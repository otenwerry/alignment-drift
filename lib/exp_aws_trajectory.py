"""AWS VM execution for real-environment trajectories.

This module can launch paid EC2 instances, so its name is exp_-prefixed. The public
pipeline imports its controller functions; EC2 user data invokes the worker entrypoint
at the bottom. Importing the module is free and does not construct AWS clients.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from aws_source_bundle import (
    SourceBundleError,
    build_source_bundle as _build_source_bundle,
    runtime_hash as _runtime_hash,
    sha256_file,
    source_file_list as _standalone_source_file_list,
)
from aws_runtime_contract import (
    AWS_SCHEMA_VERSION,
    DEFAULT_SECRET_NAMES,
    DEFAULT_WORKER_PIPELINE_SCRIPT,
    FAILURE_PACKAGE_SECONDS,
    SSM_PARAMETER_NAME,
    WORKER_ACTIVITY_LOG_PAYLOAD_PATH,
    WORKER_PIPELINE_SCRIPTS,
    WORKER_PREFIX_PAYLOAD_PATH,
    WORKER_VERSION,
)
from exp_subscription_harness import (
    CLAUDE_AUTH_B64_ENV,
    OPENCODE_GO_API_KEY_ENV,
    claude_credentials_b64,
)
from exp_target_harness import opencode_go_model_spec, production_scaffold_for_target
from inspect_task_naming import bounded_inspect_task_name
from project_paths import (
    AWS_AUTH_CACHE_ROOT,
    DATA_ROOT,
    ENV_FILE,
    LOGS_ROOT,
    REMOTE_CAMPAIGNS_ROOT,
)
from sandbox_runtime import (
    compose_file_for_harness,
    discover_sandbox_compose_files,
    sandbox_compose_relative,
    seed_family_from_path,
)


STACK_NAME = "mats-environments"
DEFAULT_REGION = "us-west-2"
DEFAULT_INSTANCE_TYPE = "c7a.xlarge"
# Verified 2026-08-14: us-west-2 Standard On-Demand quota is 1,000 vCPUs;
# c7a.xlarge uses 4 vCPUs, allowing 250 simultaneous trajectory workers.
DEFAULT_VM_CONCURRENCY = 250
LEGACY_ROOT_VOLUME_GB = 16
ROOT_VOLUME_GB = 32
S3_RETENTION_DAYS = 7
AGENT_TIME_LIMIT_SECONDS = 70 * 60
# A clean first pass can reset the agent deadline to 45 minutes from its follow-up,
# making the maximum ML agent trajectory 115 minutes. The broader worker watchdog
# remains a conservative allowance for boot, judging, artifact packaging, and upload.
UNCONDITIONAL_TERMINATION_SECONDS = 4 * 3600 + 30 * 60
AMI_BUILDER_WATCHDOG_MINUTES = 90
STANDARD_VCPU_QUOTA_CODE = "L-1216C47A"
AWS_CLI_VERSION = "2.36.2"
# The builder script itself is not part of ``runtime_dependency_files``. Bump this
# whenever its installed host runtime changes so setup cannot reuse a stale AMI.
RUNTIME_VERSION = (
    f"ubuntu24-docker-uv-environments-v7-bwrap-apparmor-node-npm-awscli-"
    f"{AWS_CLI_VERSION}"
)
MATS_APPROVED_FUNDING = "MATS-approved"
PERSONAL_REIMBURSEMENT_FUNDING = "personal-reimbursement"
SUPPORTED_FUNDING = frozenset({
    MATS_APPROVED_FUNDING,
    PERSONAL_REIMBURSEMENT_FUNDING,
})
AWS_AUTH_EXPIRY_CODES = frozenset({
    "ExpiredToken",
    "ExpiredTokenException",
    "RequestExpired",
})
EC2_LAUNCH_THROTTLE_CODES = frozenset({
    "RequestLimitExceeded",
    "RequestThrottled",
    "ServiceUnavailable",
    "Throttling",
    "ThrottlingException",
})
EC2_LAUNCH_MAX_ATTEMPTS = 12
EC2_LAUNCH_BACKOFF_CAP_SECONDS = 30.0
AWS_CREDENTIAL_PROCESS = (
    Path(__file__).resolve().parent.parent / "tools" / "aws_credential_process.py"
)

class AwsTrajectoryError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


_sha256_file = sha256_file


def source_file_list(
    repo_root: Path, environments_root: Path | None = None
) -> list[Path]:
    """Compatibility facade for the standalone AWS source-file selector."""

    root = environments_root or repo_root / "environments"
    try:
        return _standalone_source_file_list(repo_root, root)
    except SourceBundleError as error:
        raise AwsTrajectoryError(str(error)) from None


def _safe_slug(value: str, *, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-").lower()
    return (cleaned or "cell")[:limit]


def _client_error_code(ex: Exception) -> str:
    response = getattr(ex, "response", {}) or {}
    return str((response.get("Error") or {}).get("Code") or "")


def _authentication_recovery_message(state: dict, error: Exception) -> str | None:
    code = _client_error_code(error)
    if code not in AWS_AUTH_EXPIRY_CODES:
        return None
    campaign_id = str(state["campaign_id"])
    region = str(state.get("region") or DEFAULT_REGION)
    run_profile = os.environ.get("AWS_PROFILE") or "mats-run"
    # This repository's documented profile split uses mats-login for interactive
    # authentication and mats-run for SDK/CLI execution.
    login_profile = "mats-login" if run_profile == "mats-run" else run_profile
    pipeline_script = DEFAULT_WORKER_PIPELINE_SCRIPT
    cells = state.get("cells") or []
    if cells and cells[0].get("pipeline_script") in WORKER_PIPELINE_SCRIPTS:
        pipeline_script = cells[0]["pipeline_script"]
    return (
        f"AWS authentication expired while monitoring campaign {campaign_id} "
        f"({code}). The workers continue independently and no trajectory will be "
        f"relaunched. Run `aws login --profile {login_profile} --region {region}`, "
        f"verify `aws sts get-caller-identity --profile {run_profile}`, then run "
        f"`uv run {pipeline_script} --resume-campaign={campaign_id}`."
    )


def build_source_bundle(environments_root: Path, output_dir: Path) -> dict:
    try:
        return _build_source_bundle(environments_root, output_dir)
    except SourceBundleError as error:
        raise AwsTrajectoryError(str(error)) from None


def runtime_hash(environments_root: Path) -> str:
    try:
        return _runtime_hash(environments_root, RUNTIME_VERSION)
    except SourceBundleError as error:
        raise AwsTrajectoryError(str(error)) from None


def _seed_family(cfg: dict) -> str:
    configured = cfg.get("family")
    try:
        derived = seed_family_from_path(cfg["seeds_path"])
    except (KeyError, ValueError) as error:
        raise AwsTrajectoryError(str(error)) from None
    if configured is not None and configured != derived:
        raise AwsTrajectoryError(
            f"configured seed family {configured!r} does not match "
            f"--seed-dir path family {derived!r}"
        )
    return derived


def _sandbox_compose_for_target(
    cfg: dict, *, target: str, target_model: str
) -> str:
    scaffold = production_scaffold_for_target(target, target_model)
    compose_file = compose_file_for_harness(
        harness=cfg["harness"], scaffold=scaffold
    )
    try:
        return sandbox_compose_relative(_seed_family(cfg), compose_file)
    except ValueError as error:
        raise AwsTrajectoryError(str(error)) from None


def required_secret_names(model_slugs: list[str]) -> set[str]:
    required: set[str] = set()
    for slug in model_slugs:
        value = slug.lower()
        if value.startswith("openrouter/"):
            required.add("OPENROUTER_API_KEY")
        elif value.startswith("anthropic/"):
            required.add("ANTHROPIC_API_KEY")
        elif value.startswith("openai/"):
            required.add("OPENAI_API_KEY")
        elif value.startswith(("google/", "gemini/")):
            required.add("GOOGLE_API_KEY")
        elif value.startswith("mistral/"):
            required.add("MISTRAL_API_KEY")
        elif value.startswith("together/"):
            required.add("TOGETHER_API_KEY")
        elif value.startswith("groq/"):
            required.add("GROQ_API_KEY")
        elif value.startswith(("xai/", "x-ai/")):
            required.add("XAI_API_KEY")
    return required


def aws_clients(region: str) -> dict:
    import boto3
    from dotenv import load_dotenv

    # AWS controller entrypoints share the same project-level environment file as
    # agent and judge credentials. ``override=False`` preserves the normal rule that
    # a one-off shell setting wins over the saved project default.
    load_dotenv(ENV_FILE, override=False)
    profile = os.environ.get("AWS_PROFILE") or None
    direct_session = boto3.Session(profile_name=profile, region_name=region)
    if profile and shutil.which("aws") is not None:
        # Even when Botocore can initially read an ``aws login`` profile, it can
        # materialize that profile as fixed short-lived credentials.  Long-running
        # controllers then fail at expiry.  The CLI export provider preserves the
        # Expiration field and is re-invoked by Botocore when credentials need to be
        # refreshed.
        session = _aws_cli_export_session(boto3, profile=profile, region=region)
    else:
        credentials = direct_session.get_credentials()
        if credentials is None:
            raise AwsTrajectoryError(
                f"AWS_PROFILE is unset after loading {ENV_FILE}, and Boto3 found no "
                "default credentials; add AWS_PROFILE=<profile> to that exact file"
            )
        session = direct_session
        if profile:
            print(
                f"AWS credentials: profile={profile} via Boto3 credential chain "
                "(AWS CLI unavailable, so automatic CLI refresh is disabled)"
            )
    return {
        "sts": session.client("sts"),
        "s3": session.client("s3"),
        "ec2": session.client("ec2"),
        "iam": session.client("iam"),
        "ssm": session.client("ssm"),
        "quotas": session.client("service-quotas"),
        "pricing": session.client("pricing", region_name="us-east-1"),
    }


def _aws_cli_export_session(boto3_module, *, profile: str, region: str):
    """Bridge an AWS CLI profile through one shared refresh/cache process.

    Every controller has its own Botocore provider.  The helper serializes their CLI
    refreshes and reuses one protected temporary-credential cache, preventing a batch
    of long-running controllers from stampeding the AWS login token endpoint when the
    original credentials expire together.
    """

    from botocore.credentials import ProcessProvider
    from botocore.session import Session as BotocoreSession

    aws = shutil.which("aws")
    if aws is None:
        raise AwsTrajectoryError(
            f"AWS profile {profile!r} has no SDK-readable credentials and the AWS "
            "CLI is not on PATH; install AWS CLI v2 or use an SDK-readable profile"
        )
    if not AWS_CREDENTIAL_PROCESS.is_file():
        raise AwsTrajectoryError(
            f"AWS credential-process helper is missing: {AWS_CREDENTIAL_PROCESS}"
        )
    bridge_profile = "mats-aws-cli-export"
    cache_key = hashlib.sha256(profile.encode()).hexdigest()[:16]
    auth_root = AWS_AUTH_CACHE_ROOT
    command = shlex.join([
        sys.executable,
        str(AWS_CREDENTIAL_PROCESS),
        "--profile",
        profile,
        "--aws-path",
        aws,
        "--cache-path",
        str(auth_root / f"credentials-{cache_key}.json"),
        "--lock-path",
        str(auth_root / f"credentials-{cache_key}.lock"),
    ])
    provider = ProcessProvider(
        bridge_profile,
        load_config=lambda: {
            "profiles": {
                bridge_profile: {"credential_process": command},
            }
        },
    )
    botocore_session = BotocoreSession()
    botocore_session.get_component("credential_provider").insert_before(
        "env", provider
    )
    session = boto3_module.Session(
        botocore_session=botocore_session, region_name=region
    )
    try:
        credentials = session.get_credentials()
    except Exception as error:
        raise AwsTrajectoryError(
            f"AWS CLI could not export credentials for profile {profile!r}; run "
            f"`aws login --profile {profile}` and retry"
        ) from error
    if credentials is None:
        raise AwsTrajectoryError(
            f"AWS CLI exported no credentials for profile {profile!r}; run "
            f"`aws login --profile {profile}` and retry"
        )
    print(
        f"AWS credentials: profile={profile} via shared AWS CLI refresh/cache bridge"
    )
    return session


def account_id(clients: dict) -> str:
    identity = clients["sts"].get_caller_identity()
    arn = str(identity.get("Arn") or "")
    if arn.endswith(":root"):
        raise AwsTrajectoryError(
            "refusing AWS root-user credentials; use an IAM role, IAM Identity "
            "Center profile, or non-root IAM user"
        )
    return str(identity["Account"])


def default_bucket_name(account: str, region: str) -> str:
    return f"{STACK_NAME}-{account}-{region}"


def _tags_dict(tags: list[dict]) -> dict[str, str]:
    return {str(tag["Key"]): str(tag["Value"]) for tag in tags}


def ensure_bucket(clients: dict, *, bucket: str, region: str, funding: str) -> None:
    if funding not in SUPPORTED_FUNDING:
        raise AwsTrajectoryError(f"unsupported AWS funding label: {funding!r}")
    s3 = clients["s3"]
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception as ex:
        if _client_error_code(ex) not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        args: dict[str, Any] = {"Bucket": bucket}
        if region != "us-east-1":
            args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**args)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={"Rules": [{
            "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
            "BucketKeyEnabled": True,
        }]},
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={"Rules": [{
            "ID": "delete-remote-trajectories-after-seven-days",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Expiration": {"Days": S3_RETENTION_DAYS},
            "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
        }]},
    )
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": [
        {"Key": "Project", "Value": STACK_NAME},
        {"Key": "Funding", "Value": funding},
        {"Key": "ManagedBy", "Value": AWS_SCHEMA_VERSION},
    ]})


def _iam_policy(account: str, bucket: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": [
                    f"arn:aws:s3:::{bucket}/campaigns/*",
                    f"arn:aws:s3:::{bucket}/setup/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": ["campaigns/*", "setup/*"]}},
            },
            {
                "Effect": "Allow",
                "Action": "ssm:GetParameter",
                "Resource": (
                    f"arn:aws:ssm:*:{account}:parameter"
                    f"{SSM_PARAMETER_NAME}"
                ),
            },
        ],
    }


def ensure_worker_role(clients: dict, *, account: str, bucket: str) -> str:
    iam = clients["iam"]
    role_name = f"{STACK_NAME}-worker"
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        iam.get_role(RoleName=role_name)
    except Exception as ex:
        if _client_error_code(ex) != "NoSuchEntity":
            raise
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="One isolated VM per MATS real-environment trajectory",
            Tags=[{"Key": "Project", "Value": STACK_NAME}],
        )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{STACK_NAME}-worker-access",
        PolicyDocument=json.dumps(_iam_policy(account, bucket)),
    )
    profile_name = role_name
    try:
        profile = iam.get_instance_profile(InstanceProfileName=profile_name)
    except Exception as ex:
        if _client_error_code(ex) != "NoSuchEntity":
            raise
        profile = iam.create_instance_profile(InstanceProfileName=profile_name)
    roles = (profile.get("InstanceProfile") or {}).get("Roles") or []
    if not any(role.get("RoleName") == role_name for role in roles):
        try:
            iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name, RoleName=role_name
            )
        except Exception as ex:
            if _client_error_code(ex) not in {"LimitExceeded", "EntityAlreadyExists"}:
                raise
    return profile_name


def ensure_security_group(clients: dict) -> tuple[str, str]:
    ec2 = clients["ec2"]
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])[
        "Vpcs"
    ]
    if not vpcs:
        raise AwsTrajectoryError("AWS region has no default VPC; setup will not guess a network")
    vpc_id = vpcs[0]["VpcId"]
    name = f"{STACK_NAME}-no-ingress"
    groups = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [name]},
        {"Name": "vpc-id", "Values": [vpc_id]},
    ])["SecurityGroups"]
    if groups:
        return groups[0]["GroupId"], vpc_id
    group_id = ec2.create_security_group(
        GroupName=name,
        Description="MATS trajectory workers: no inbound connections",
        VpcId=vpc_id,
        TagSpecifications=[{"ResourceType": "security-group", "Tags": [
            {"Key": "Project", "Value": STACK_NAME},
        ]}],
    )["GroupId"]
    return group_id, vpc_id


def _default_subnet(clients: dict, vpc_id: str, instance_type: str) -> str:
    subnets = clients["ec2"].describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]},
        {"Name": "default-for-az", "Values": ["true"]},
    ])["Subnets"]
    if not subnets:
        raise AwsTrajectoryError("default VPC has no default subnet")
    offerings = clients["ec2"].describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    )["InstanceTypeOfferings"]
    available_zones = {offering["Location"] for offering in offerings}
    supported = [
        subnet for subnet in subnets
        if subnet.get("AvailabilityZone") in available_zones
    ]
    if not supported:
        raise AwsTrajectoryError(
            f"{instance_type} is not offered in any default subnet in this region"
        )
    return sorted(supported, key=lambda subnet: subnet["AvailabilityZone"])[0]["SubnetId"]


def put_api_keys(
    clients: dict,
    *,
    extra_names: list[str],
    ship_claude_subscription_login: bool = False,
) -> list[str]:
    allowed = list(dict.fromkeys([*DEFAULT_SECRET_NAMES, *extra_names]))
    invalid = [name for name in allowed if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)]
    if invalid:
        raise AwsTrajectoryError(
            "API-key allow-list contains invalid environment names: "
            + ", ".join(invalid)
        )
    missing_explicit = [name for name in extra_names if not os.environ.get(name)]
    if missing_explicit:
        raise AwsTrajectoryError(
            "explicit --aws-secret-env values are not set locally: "
            + ", ".join(missing_explicit)
        )
    values = {name: os.environ[name] for name in allowed if os.environ.get(name)}
    if (
        ship_claude_subscription_login
        and "CLAUDE_CODE_OAUTH_TOKEN" not in values
        and CLAUDE_AUTH_B64_ENV not in values
    ):
        # A host signed in via ~/.claude/.credentials.json or the macOS keychain
        # has no shippable env var; convert that login so subscription workers
        # can use it. Only a subscription-harness setup opts into this.
        encoded = claude_credentials_b64()
        if encoded:
            values[CLAUDE_AUTH_B64_ENV] = encoded
            print(
                "AWS setup: shipping the host Claude Code subscription login as "
                f"{CLAUDE_AUTH_B64_ENV} (no CLAUDE_CODE_OAUTH_TOKEN set)."
            )
    if not values:
        raise AwsTrajectoryError(
            "none of the approved model API-key environment variables are set"
        )
    exists = True
    existing_tier = None
    try:
        clients["ssm"].get_parameter(Name=SSM_PARAMETER_NAME)
    except Exception as ex:
        if _client_error_code(ex) != "ParameterNotFound":
            raise
        exists = False
    if exists:
        # GetParameter does not return Tier. DescribeParameters does, and an
        # existing Advanced parameter cannot be overwritten as Standard even
        # when its new value happens to fit under the Standard 4 KiB limit.
        metadata = clients["ssm"].describe_parameters(
            ParameterFilters=[{
                "Key": "Name",
                "Option": "Equals",
                "Values": [SSM_PARAMETER_NAME],
            }],
            MaxResults=1,
        )
        parameters = metadata.get("Parameters") or []
        matching = [
            parameter for parameter in parameters
            if parameter.get("Name") == SSM_PARAMETER_NAME
        ]
        if not matching:
            raise AwsTrajectoryError(
                "existing AWS worker-secret parameter was not returned by "
                "DescribeParameters"
            )
        existing_tier = matching[0].get("Tier")
    serialized = json.dumps(values, sort_keys=True)
    serialized_bytes = len(serialized.encode())
    # Standard parameters are limited to 4 KiB. Advanced parameters are billable
    # shared infrastructure and cannot be downgraded, so preserve an existing tier.
    tier = (
        "Advanced"
        if serialized_bytes > 4096 or existing_tier == "Advanced"
        else "Standard"
    )
    if serialized_bytes > 8192:
        raise AwsTrajectoryError(
            "encrypted AWS worker secrets exceed SSM's 8 KiB Advanced-parameter "
            "limit; prefer CODEX_SUBSCRIPTION_AUTH_JSON_GZIP_B64 or remove unused keys"
        )
    if tier == "Advanced":
        print(
            "AWS setup warning: the encrypted worker-secret bundle uses a billable "
            "SSM Advanced parameter because it exceeds 4 KiB (or was already Advanced)."
        )
    put_args = dict(
        Name=SSM_PARAMETER_NAME,
        Description="Allow-listed model API keys for MATS trajectory VMs",
        Type="SecureString",
        Value=serialized,
        Tier=tier,
    )
    if exists:
        put_args["Overwrite"] = True
    else:
        put_args["Tags"] = [{"Key": "Project", "Value": STACK_NAME}]
    clients["ssm"].put_parameter(**put_args)
    return sorted(values)


def _find_runtime_ami(clients: dict, wanted_hash: str) -> str | None:
    images = clients["ec2"].describe_images(
        Owners=["self"],
        Filters=[
            {"Name": "state", "Values": ["available"]},
            {"Name": "tag:Project", "Values": [STACK_NAME]},
            {"Name": "tag:RuntimeHash", "Values": [wanted_hash]},
        ],
    )["Images"]
    if not images:
        return None
    images.sort(key=lambda image: image.get("CreationDate", ""), reverse=True)
    return images[0]["ImageId"]


def _builder_user_data(*, bucket: str, region: str, source_key: str,
                       source_sha: str, runtime: str) -> str:
    ready_key = f"setup/{runtime}/ready.json"
    log_key = f"setup/{runtime}/builder.log"
    source_uri = shlex.quote(f"s3://{bucket}/{source_key}")
    ready_uri = shlex.quote(f"s3://{bucket}/{ready_key}")
    log_uri = shlex.quote(f"s3://{bucket}/{log_key}")
    quoted_region = shlex.quote(region)
    checksum_line = shlex.quote(f"{source_sha}  /tmp/mats-source.tar.gz")
    dollar = "$"
    return f"""#!/bin/bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
log_file=/var/log/mats-ami-builder.log
touch "$log_file"
exec > >(tee -a "$log_file") 2>&1
stage=bootstrapping

upload_log() {{
  aws s3 cp "$log_file" {log_uri} --sse AES256 --region {quoted_region} >/dev/null || true
}}

write_status() {{
  stage="$1"
  upload_log
  printf '{{"runtime_hash":"{runtime}","state":"%s","updated_at":"%s"}}' \
    "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >/tmp/ready.json
  aws s3 cp /tmp/ready.json {ready_uri} --sse AES256 --region {quoted_region} >/dev/null
}}

on_error() {{
  exit_code=$?
  trap - ERR
  set +e
  echo "AMI builder failed in stage=$stage exit_code=$exit_code"
  if command -v aws >/dev/null 2>&1; then
    upload_log
    printf '{{"runtime_hash":"{runtime}","state":"failed","stage":"%s","exit_code":%d,"failed_at":"%s"}}' \
      "$stage" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >/tmp/ready.json
    aws s3 cp /tmp/ready.json {ready_uri} --sse AES256 --region {quoted_region} >/dev/null
  fi
  shutdown -h now
  exit "$exit_code"
}}
trap on_error ERR

# This timer survives the reboot used to make the AMI. It only powers off this exact
# builder: worker VMs inherit the files but have a different hardware UUID and no-op.
shutdown -h +{AMI_BUILDER_WATCHDOG_MINUTES}
read -r builder_uuid </sys/devices/virtual/dmi/id/product_uuid
builder_deadline="$(date -u -d '+{AMI_BUILDER_WATCHDOG_MINUTES} minutes' '+%Y-%m-%d %H:%M:%S UTC')"
cat >/usr/local/sbin/mats-ami-builder-watchdog <<EOF
#!/bin/bash
set -eu
read -r current_uuid </sys/devices/virtual/dmi/id/product_uuid
if [ "\\{dollar}current_uuid" = "$builder_uuid" ]; then
  systemctl poweroff
fi
EOF
chmod 0755 /usr/local/sbin/mats-ami-builder-watchdog
cat >/etc/systemd/system/mats-ami-builder-watchdog.service <<'EOF'
[Unit]
Description=Terminate an abandoned MATS AMI builder

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/mats-ami-builder-watchdog
EOF
cat >/etc/systemd/system/mats-ami-builder-watchdog.timer <<EOF
[Unit]
Description=Hard lifetime cap for the MATS AMI builder

[Timer]
OnCalendar=$builder_deadline
Persistent=true
Unit=mats-ami-builder-watchdog.service

[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now mats-ami-builder-watchdog.timer
stage=installing-system-packages
apt-get update
apt-get install -y apparmor ca-certificates curl unzip docker.io docker-compose-v2 nodejs npm
node --version
npm --version

# Ubuntu 24.04 blocks unprivileged user namespaces unless the program has a narrow
# AppArmor rule. Claude Code's documented bwrap rule applies to bwrap itself, not to
# the commands that bwrap launches.
stage=configuring-bubblewrap-apparmor
cat >/etc/apparmor.d/bwrap <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {{
  userns,
  include if exists <local/bwrap>
}}
EOF
systemctl reload apparmor

# Ubuntu 24.04 removed its awscli package. Install a pinned official AWS CLI v2
# release instead; workers need it for their S3 handoff.
stage=installing-aws-cli
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64-{AWS_CLI_VERSION}.zip \
  -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli
aws --version

write_status starting-docker
systemctl enable --now docker
write_status installing-uv
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
write_status downloading-source
mkdir -p /opt/supermats
aws s3 cp {source_uri} /tmp/mats-source.tar.gz --region {quoted_region}
echo {checksum_line} | sha256sum -c -
tar -xzf /tmp/mats-source.tar.gz -C /opt/supermats
write_status syncing-python-runtime
export UV_PROJECT_ENVIRONMENT=/opt/environments-venv
uv sync --project /opt/supermats/mats/environments --frozen
write_status building-docker-images
mapfile -t compose_files < <(
  find /opt/supermats/mats/environments/sandbox -mindepth 2 -maxdepth 2 \
    -type f \\( -name compose.yaml -o -name compose.subscription.yaml \\) -print | sort
)
if [ "{dollar}{{#compose_files[@]}}" -eq 0 ]; then
  echo "No real-environment sandbox compose files found"
  exit 1
fi
for compose_file in "{dollar}{{compose_files[@]}}"; do
  echo "Building sandbox declared by {dollar}compose_file"
  docker compose -f "{dollar}compose_file" build
  if [[ "{dollar}compose_file" == *.subscription.yaml ]]; then
    echo "Checking nested Bubblewrap in {dollar}compose_file"
    docker compose -f "{dollar}compose_file" run --rm --no-deps default \
      bwrap --die-with-parent --unshare-user --uid 0 --gid 0 --unshare-pid \
      --unshare-uts --unshare-ipc --unshare-cgroup-try --ro-bind / / \
      --dev-bind /dev /dev --ro-bind /proc /proc /usr/bin/true
    docker compose -f "{dollar}compose_file" down
  fi
done
mkdir -p /opt/mats-runtime
echo '{runtime}' > /opt/mats-runtime/runtime-hash
upload_log
printf '{{"runtime_hash":"{runtime}","state":"ready","ready_at":"%s"}}' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >/tmp/ready.json
aws s3 cp /tmp/ready.json {ready_uri} --sse AES256 --region {quoted_region}
"""


def build_runtime_ami(clients: dict, *, region: str, instance_type: str,
                      profile: str, security_group: str, subnet: str, bucket: str,
                      bundle: dict, wanted_hash: str) -> str:
    existing = _find_runtime_ami(clients, wanted_hash)
    if existing:
        print(f"  AMI runtime is current: {existing}")
        return existing
    source_key = f"setup/{wanted_hash}/source.tar.gz"
    clients["s3"].upload_file(bundle["path"], bucket, source_key)
    ubuntu = clients["ssm"].get_parameter(
        Name=("/aws/service/canonical/ubuntu/server/24.04/stable/current/"
              "amd64/hvm/ebs-gp3/ami-id")
    )["Parameter"]["Value"]
    user_data = _builder_user_data(
        bucket=bucket, region=region, source_key=source_key, source_sha=bundle["sha256"],
        runtime=wanted_hash,
    )
    ready_key = f"setup/{wanted_hash}/ready.json"
    log_key = f"setup/{wanted_hash}/builder.log"
    clients["s3"].put_object(
        Bucket=bucket,
        Key=ready_key,
        Body=_json_bytes({"runtime_hash": wanted_hash, "state": "booting"}),
        ServerSideEncryption="AES256",
    )
    run_args = dict(
        ImageId=ubuntu,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        ClientToken=hashlib.sha256(
            f"ami-builder:{wanted_hash}:{_utc_now()}".encode()
        ).hexdigest(),
        IamInstanceProfile={"Name": profile},
        NetworkInterfaces=[{
            "DeviceIndex": 0,
            "SubnetId": subnet,
            "Groups": [security_group],
            "AssociatePublicIpAddress": True,
        }],
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": ROOT_VOLUME_GB,
                "VolumeType": "gp3",
                "Encrypted": True,
                "DeleteOnTermination": True,
            },
        }],
        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=user_data,
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"{STACK_NAME}-ami-builder"},
            {"Key": "Project", "Value": STACK_NAME},
            {"Key": "Role", "Value": "ami-builder"},
            {"Key": "RuntimeHash", "Value": wanted_hash},
        ]}],
    )
    deadline = time.monotonic() + 2 * 60
    while True:
        try:
            response = clients["ec2"].run_instances(**run_args)
            break
        except Exception as ex:
            profile_not_ready = (
                _client_error_code(ex) == "InvalidParameterValue"
                and "instance profile" in str(ex).lower()
            )
            if not profile_not_ready or time.monotonic() >= deadline:
                raise
            print("  waiting for the new EC2 instance profile to propagate")
            time.sleep(5)
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"  building reusable runtime AMI on {instance_id} (this can take a while)")
    deadline = time.monotonic() + 45 * 60
    ready = False
    last_stage = None
    last_progress_print = time.monotonic()
    completed = False
    try:
        while time.monotonic() < deadline:
            details = clients["ec2"].describe_instances(InstanceIds=[instance_id])
            state = details["Reservations"][0]["Instances"][0]["State"]["Name"]
            marker = _s3_json_or_none(clients["s3"], bucket, ready_key)
            marker_matches = bool(
                marker and marker.get("runtime_hash") == wanted_hash
            )
            build_stage = marker.get("state") if marker_matches else None
            if build_stage and build_stage != last_stage:
                print(f"  AMI builder stage: {build_stage}")
                last_stage = build_stage
                last_progress_print = time.monotonic()
            elif time.monotonic() - last_progress_print >= 120:
                print(f"  AMI builder still working: {last_stage or state}")
                last_progress_print = time.monotonic()
            if marker_matches and build_stage == "failed":
                failed_stage = marker.get("stage") or "unknown"
                exit_code = marker.get("exit_code")
                raise AwsTrajectoryError(
                    f"AMI builder failed during {failed_stage} (exit {exit_code}); "
                    f"log: s3://{bucket}/{log_key}"
                )
            ready = bool(
                marker_matches
                and marker.get("ready_at")
            )
            if state == "running" and ready:
                break
            if state in {"stopped", "terminated", "shutting-down"}:
                raise AwsTrajectoryError(
                    "AMI builder terminated before becoming ready; "
                    f"last stage={last_stage or 'unknown'}, log=s3://{bucket}/{log_key}"
                )
            time.sleep(15)
        else:
            raise AwsTrajectoryError(
                "AMI builder did not finish within 45 minutes; "
                f"last stage={last_stage or 'unknown'}, log=s3://{bucket}/{log_key}"
            )
        image = clients["ec2"].create_image(
            InstanceId=instance_id,
            Name=f"{STACK_NAME}-{wanted_hash[:12]}",
            Description=f"Pinned MATS real-environment runtime {wanted_hash}",
            TagSpecifications=[{"ResourceType": "image", "Tags": [
                {"Key": "Project", "Value": STACK_NAME},
                {"Key": "RuntimeHash", "Value": wanted_hash},
                {"Key": "ManagedBy", "Value": AWS_SCHEMA_VERSION},
            ]}],
        )["ImageId"]
        clients["ec2"].get_waiter("image_available").wait(
            ImageIds=[image], WaiterConfig={"Delay": 15, "MaxAttempts": 120}
        )
        print(f"  created runtime AMI {image}")
        completed = True
        return image
    finally:
        try:
            clients["ec2"].terminate_instances(InstanceIds=[instance_id])
        except Exception as cleanup_error:
            if completed:
                raise
            print(
                "  WARNING: controller could not terminate failed AMI builder "
                f"{instance_id}: {cleanup_error}. Its independent "
                f"{AMI_BUILDER_WATCHDOG_MINUTES}-minute watchdog remains active."
            )


def ensure_launch_template(clients: dict, *, ami: str, instance_type: str,
                           profile: str, security_group: str, subnet: str) -> str:
    ec2 = clients["ec2"]
    name = f"{STACK_NAME}-trajectory"
    data = {
        "ImageId": ami,
        "InstanceType": instance_type,
        "IamInstanceProfile": {"Name": profile},
        "NetworkInterfaces": [{
            "DeviceIndex": 0,
            "SubnetId": subnet,
            "Groups": [security_group],
            "AssociatePublicIpAddress": True,
        }],
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": ROOT_VOLUME_GB,
                "VolumeType": "gp3",
                "Encrypted": True,
                "DeleteOnTermination": True,
            },
        }],
        "MetadataOptions": {
            "HttpTokens": "required", "HttpEndpoint": "enabled",
            "InstanceMetadataTags": "disabled",
        },
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [{"ResourceType": "volume", "Tags": [
            {"Key": "Project", "Value": STACK_NAME},
        ]}],
    }
    try:
        template = ec2.describe_launch_templates(LaunchTemplateNames=[name])[
            "LaunchTemplates"
        ][0]
    except Exception as ex:
        if _client_error_code(ex) not in {"InvalidLaunchTemplateName.NotFoundException",
                                         "InvalidLaunchTemplateId.NotFound"}:
            raise
        created = ec2.create_launch_template(
            LaunchTemplateName=name,
            VersionDescription=RUNTIME_VERSION,
            LaunchTemplateData=data,
            TagSpecifications=[{"ResourceType": "launch-template", "Tags": [
                {"Key": "Project", "Value": STACK_NAME},
            ]}],
        )["LaunchTemplate"]
        return created["LaunchTemplateId"]
    version = ec2.create_launch_template_version(
        LaunchTemplateId=template["LaunchTemplateId"],
        VersionDescription=RUNTIME_VERSION,
        LaunchTemplateData=data,
    )["LaunchTemplateVersion"]["VersionNumber"]
    ec2.modify_launch_template(
        LaunchTemplateId=template["LaunchTemplateId"],
        DefaultVersion=str(version),
    )
    return template["LaunchTemplateId"]


def _validate_worker_shape(clients: dict, instance_type: str) -> tuple[int, int]:
    info = clients["ec2"].describe_instance_types(
        InstanceTypes=[instance_type]
    )["InstanceTypes"][0]
    vcpus = int(info["VCpuInfo"]["DefaultVCpus"])
    memory_mib = int(info["MemoryInfo"]["SizeInMiB"])
    if vcpus != 4 or memory_mib < 8192:
        raise AwsTrajectoryError(
            f"{instance_type} is {vcpus} vCPU/{memory_mib} MiB; expected "
            "the pinned 4-vCPU, >=8-GB worker shape"
        )
    return vcpus, memory_mib


def _launch_template_root_volume_gb(template_data: dict) -> int | None:
    for mapping in template_data.get("BlockDeviceMappings") or []:
        if mapping.get("DeviceName") != "/dev/sda1":
            continue
        size = (mapping.get("Ebs") or {}).get("VolumeSize")
        try:
            return int(size)
        except (TypeError, ValueError):
            return None
    return None


def setup_aws(cfg: dict, environments_root: Path) -> dict:
    approved = bool(cfg.get("confirm_approved_account"))
    personal = bool(cfg.get("confirm_personal_account"))
    if approved == personal:
        raise AwsTrajectoryError(
            "--aws-setup requires exactly one of --confirm-approved-account or "
            "--confirm-personal-account"
        )
    funding = MATS_APPROVED_FUNDING if approved else PERSONAL_REIMBURSEMENT_FUNDING
    region = cfg["aws_region"]
    clients = aws_clients(region)
    account = account_id(clients)
    bucket = cfg.get("aws_bucket") or default_bucket_name(account, region)
    print(
        f"AWS setup: account={account} region={region} bucket={bucket} "
        f"funding={funding}"
    )
    _validate_worker_shape(clients, cfg["aws_instance_type"])
    ensure_bucket(clients, bucket=bucket, region=region, funding=funding)
    secret_names = put_api_keys(
        clients,
        extra_names=cfg.get("aws_secret_env") or [],
        ship_claude_subscription_login=cfg.get("harness") == "subscription",
    )
    profile = ensure_worker_role(clients, account=account, bucket=bucket)
    security_group, vpc_id = ensure_security_group(clients)
    subnet = _default_subnet(clients, vpc_id, cfg["aws_instance_type"])
    with tempfile.TemporaryDirectory(prefix="mats-aws-setup-") as tmp:
        bundle = build_source_bundle(environments_root, Path(tmp))
        wanted_hash = runtime_hash(environments_root)
        ami = build_runtime_ami(
            clients,
            region=region,
            instance_type=cfg["aws_instance_type"],
            profile=profile,
            security_group=security_group,
            subnet=subnet,
            bucket=bucket,
            bundle=bundle,
            wanted_hash=wanted_hash,
        )
    template = ensure_launch_template(
        clients,
        ami=ami,
        instance_type=cfg["aws_instance_type"],
        profile=profile,
        security_group=security_group,
        subnet=subnet,
    )
    result = {
        "account": account,
        "region": region,
        "bucket": bucket,
        "funding": funding,
        "secret_names": secret_names,
        "ami": ami,
        "runtime_hash": wanted_hash,
        "launch_template": template,
        "root_volume_gb": ROOT_VOLUME_GB,
    }
    print("AWS setup complete: " + json.dumps(result, sort_keys=True))
    return result


def instance_hourly_price(clients: dict, *, region: str, instance_type: str) -> float:
    products = clients["pricing"].get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=100,
    )["PriceList"]
    prices = []
    for raw in products:
        product = json.loads(raw) if isinstance(raw, str) else raw
        for term in (product.get("terms") or {}).get("OnDemand", {}).values():
            for dimension in (term.get("priceDimensions") or {}).values():
                if dimension.get("unit") == "Hrs":
                    prices.append(float(dimension["pricePerUnit"]["USD"]))
    prices = sorted(set(price for price in prices if price > 0))
    if len(prices) != 1:
        raise AwsTrajectoryError(
            f"expected one on-demand Linux price for {instance_type} in {region}; got {prices}"
        )
    return prices[0]


def preflight_aws(
    cfg: dict,
    environments_root: Path,
    *,
    model_slugs: list[str],
    required_secrets: set[str] | None = None,
    required_secret_alternatives: list[tuple[str, ...]] | None = None,
) -> dict:
    region = cfg["aws_region"]
    clients = aws_clients(region)
    account = account_id(clients)
    bucket = cfg.get("aws_bucket") or default_bucket_name(account, region)
    try:
        tags = _tags_dict(clients["s3"].get_bucket_tagging(Bucket=bucket)["TagSet"])
    except Exception as ex:
        raise AwsTrajectoryError(f"AWS setup bucket is unavailable: {bucket}") from ex
    funding = tags.get("Funding")
    if funding not in SUPPORTED_FUNDING:
        raise AwsTrajectoryError(
            "AWS bucket has no recognized team-funded or personal/reimbursement label"
        )
    secret_value = clients["ssm"].get_parameter(
        Name=SSM_PARAMETER_NAME, WithDecryption=True
    )["Parameter"]["Value"]
    stored_secrets = json.loads(secret_value)
    if not isinstance(stored_secrets, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in stored_secrets.items()
    ):
        raise AwsTrajectoryError("SSM API-key parameter has the wrong shape; rerun setup")
    stored_names = set(stored_secrets)
    missing = (required_secret_names(model_slugs) | (required_secrets or set())) - stored_names
    if missing:
        raise AwsTrajectoryError(
            "SSM is missing required API keys: " + ", ".join(sorted(missing))
        )
    unmet_alternatives = [
        group
        for group in (required_secret_alternatives or [])
        if not stored_names.intersection(group)
    ]
    if unmet_alternatives:
        choices = [" or ".join(group) for group in unmet_alternatives]
        raise AwsTrajectoryError(
            "SSM is missing required subscription authentication: "
            + "; ".join(choices)
        )
    vcpus, memory_mib = _validate_worker_shape(clients, cfg["aws_instance_type"])
    wanted_hash = runtime_hash(environments_root)
    ami = _find_runtime_ami(clients, wanted_hash)
    if not ami:
        raise AwsTrajectoryError("runtime AMI is missing or stale; run --aws-setup")
    templates = clients["ec2"].describe_launch_templates(
        LaunchTemplateNames=[f"{STACK_NAME}-trajectory"]
    )["LaunchTemplates"]
    if not templates:
        raise AwsTrajectoryError("trajectory launch template is missing; run --aws-setup")
    template = templates[0]
    version = clients["ec2"].describe_launch_template_versions(
        LaunchTemplateId=template["LaunchTemplateId"],
        Versions=[str(template["DefaultVersionNumber"])],
    )["LaunchTemplateVersions"][0]["LaunchTemplateData"]
    template_volume_gb = _launch_template_root_volume_gb(version)
    if (
        version.get("ImageId") != ami
        or version.get("InstanceType") != cfg["aws_instance_type"]
        or template_volume_gb != ROOT_VOLUME_GB
    ):
        detail = (
            f"expected image={ami}, instance={cfg['aws_instance_type']}, "
            f"root={ROOT_VOLUME_GB} GB; found image={version.get('ImageId')}, "
            f"instance={version.get('InstanceType')}, root={template_volume_gb} GB"
        )
        raise AwsTrajectoryError(
            f"trajectory launch template is stale ({detail}); run --aws-setup"
        )
    quota = float(clients["quotas"].get_service_quota(
        ServiceCode="ec2", QuotaCode=STANDARD_VCPU_QUOTA_CODE
    )["Quota"]["Value"])
    needed_vcpus = vcpus * int(cfg["vm_concurrency"])
    if quota < needed_vcpus:
        raise AwsTrajectoryError(
            f"on-demand Standard quota is {quota:g} vCPUs; {cfg['vm_concurrency']} "
            f"workers require {needed_vcpus}. Request the quota before spending."
        )
    price = instance_hourly_price(
        clients, region=region, instance_type=cfg["aws_instance_type"]
    )
    return {
        "clients": clients,
        "account": account,
        "bucket": bucket,
        "funding": funding,
        "ami": ami,
        "runtime_hash": wanted_hash,
        "launch_template_id": template["LaunchTemplateId"],
        "root_volume_gb": ROOT_VOLUME_GB,
        "hourly_price_usd": price,
        "vcpus": vcpus,
        "memory_mib": memory_mib,
        "quota_vcpus": quota,
        "required_vcpus": needed_vcpus,
        # The setup command may include project-specific provider keys. Jobs receive
        # the names stored in SSM during every preflight, so later runs and explicit
        # retries do not have to repeat --aws-secret-env. Values never enter a job.
        "stored_secret_names": sorted(stored_names),
    }


def build_cells(cfg: dict, *, campaign_id: str, source: dict,
                bucket: str, hourly_price: float) -> list[dict]:
    cells = []
    selections = cfg.get("_cell_selections") or [
        (target, seed, original_epoch)
        for target in cfg["targets"]
        for seed in cfg["seeds"]
        for original_epoch in range(1, cfg["epochs"] + 1)
    ]
    target_models = dict(zip(cfg["targets"], cfg["target_models"], strict=True))
    for target, seed, original_epoch in selections:
        raw = f"{target}-{seed}-e{original_epoch}"
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
        cell_id = f"{_safe_slug(raw)}-{suffix}"
        task_suffix = cell_id
        worker_run_dir = f"{campaign_id}-{cell_id}"
        cell_prefix = f"campaigns/{campaign_id}/cells/{cell_id}"
        task_name = bounded_inspect_task_name(
            f"real_audit_{target_models[target].split('/')[-1]}_{seed}",
            task_suffix,
        )
        args = [
            f"--targets={target}",
            f"--seed-dir={Path(cfg['seeds_path']).name}",
            f"--seeds={seed}",
            "--epochs=1",
            f"--reasoning={'yes' if cfg['reasoning'] else 'no'}",
            f"--harness={cfg['harness']}",
            f"--condition={cfg['condition']}",
            f"--judge={cfg['judge_resolved']}",
            f"--gate-model={cfg['gate_model']}",
            "--concurrency=1",
            "--sandbox-concurrency=1",
            f"--time-limit={cfg['time_limit']}",
            "--compute=local",
            "--skip-viewer",
        ]
        if cfg.get("pressure"):
            args.append(f"--pressure={cfg['pressure']}")
        sandbox_compose = _sandbox_compose_for_target(
            cfg, target=target, target_model=target_models[target]
        )
        cells.append({
            "schema_version": AWS_SCHEMA_VERSION,
            "kind": "trajectory",
            "campaign_id": campaign_id,
            "cell_id": cell_id,
            "target": target,
            "target_model": target_models[target],
            "harness": cfg["harness"],
            "family": _seed_family(cfg),
            "sandbox_compose": sandbox_compose,
            "seed": seed,
            "original_epoch": original_epoch,
            "task_suffix": task_suffix,
            "task_name": task_name,
            "worker_run_dir": worker_run_dir,
            "bucket": bucket,
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "funding": cfg.get("aws_funding"),
            "hourly_price_usd": hourly_price,
            "root_volume_gb": ROOT_VOLUME_GB,
            "source_key": f"campaigns/{campaign_id}/source/source.tar.gz",
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "job_key": f"{cell_prefix}/job.json",
            "result_key": f"{cell_prefix}/result.tar.gz",
            "complete_key": f"{cell_prefix}/complete.json",
            "failure_key": f"{cell_prefix}/failure.json",
            "pipeline_args": args,
            "allowed_secret_names": list(cfg.get("worker_allowed_secret_names") or []),
            "status": "planned",
        })
    return cells


def build_prefix_generation_cells(
    cfg: dict,
    *,
    campaign_id: str,
    source: dict,
    bucket: str,
    hourly_price: float,
) -> list[dict]:
    """Cells for no-judge prefix generation: one VM per prefix."""

    cells = []
    pipeline_script = str(
        cfg.get("prefix_pipeline_script") or "prefixes/exp_ml_prefix.py"
    )
    if pipeline_script not in WORKER_PIPELINE_SCRIPTS:
        raise AwsTrajectoryError(
            f"unsupported prefix pipeline script {pipeline_script!r}"
        )
    task_namespace = str(cfg.get("prefix_task_namespace") or "ml_prefix_only")
    campaign_namespace = str(cfg.get("prefix_campaign_namespace") or "ml-prefix")
    selections = cfg.get("_cell_selections") or [
        (target, seed, original_epoch)
        for target in cfg["targets"]
        for seed in cfg["seeds"]
        for original_epoch in range(1, cfg["epochs"] + 1)
    ]
    target_models = dict(zip(cfg["targets"], cfg["target_models"], strict=True))
    for target, seed, original_epoch in selections:
        raw = f"{campaign_namespace}-{cfg['name']}-{target}-{seed}-e{original_epoch}"
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
        cell_id = f"{_safe_slug(raw)}-{suffix}"
        task_suffix = cell_id
        worker_run_dir = f"{campaign_id}-{cell_id}"
        cell_prefix = f"campaigns/{campaign_id}/cells/{cell_id}"
        target_fragment = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")
        task_name = bounded_inspect_task_name(
            f"{task_namespace}_{seed}_{target_fragment or 'model'}",
            task_suffix,
        )
        args = [
            f"--targets={target}",
            f"--seeds={seed}",
            "--epochs=1",
            f"--harness={cfg['harness']}",
            f"--reasoning={'yes' if cfg['reasoning'] else 'no'}",
            f"--name={cfg['name']}",
            "--concurrency=1",
            "--sandbox-concurrency=1",
            f"--time-limit={cfg['time_limit']}",
            "--compute=local",
            "--skip-viewer",
        ]
        if cfg.get("pressure") is not None:
            args.append(f"--pressure={cfg['pressure']}")
        sandbox_compose = _sandbox_compose_for_target(
            cfg, target=target, target_model=target_models[target]
        )
        cells.append({
            "schema_version": AWS_SCHEMA_VERSION,
            "kind": "trajectory",
            "campaign_id": campaign_id,
            "cell_id": cell_id,
            "target": target,
            "target_model": target_models[target],
            "harness": cfg["harness"],
            "family": _seed_family(cfg),
            "sandbox_compose": sandbox_compose,
            "seed": seed,
            "original_epoch": original_epoch,
            "pipeline_script": pipeline_script,
            "task_suffix": task_suffix,
            "task_name": task_name,
            "worker_run_dir": worker_run_dir,
            "bucket": bucket,
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "funding": cfg.get("aws_funding"),
            "hourly_price_usd": hourly_price,
            "root_volume_gb": ROOT_VOLUME_GB,
            "source_key": f"campaigns/{campaign_id}/source/source.tar.gz",
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "job_key": f"{cell_prefix}/job.json",
            "result_key": f"{cell_prefix}/result.tar.gz",
            "complete_key": f"{cell_prefix}/complete.json",
            "failure_key": f"{cell_prefix}/failure.json",
            "pipeline_args": args,
            "allowed_secret_names": list(
                cfg.get("worker_allowed_secret_names") or []
            ),
            "status": "planned",
        })
    return cells


def build_continuation_cells(cfg: dict, *, campaign_id: str, source: dict,
                             bucket: str, hourly_price: float) -> list[dict]:
    """Cells for a continuation campaign: one VM per (prefix, seed, epoch).

    ``cfg["continuation"]`` carries the treatment plus one payload descriptor per
    prefix ({name, sha256, file_sha256, local_path, target, target_model,
    reasoning});
    ``file_sha256`` is the exact stored file's byte hash used for transport
    verification, while ``sha256`` is the payload's canonical identity. The payload
    file is uploaded once per campaign and downloaded by each worker to
    ``WORKER_PREFIX_PAYLOAD_PATH`` before the pipeline starts.
    """

    continuation = cfg["continuation"]
    treatment = continuation["treatment"]
    payloads_by_name = {item["name"]: item for item in continuation["payloads"]}
    selections = cfg.get("_cell_selections") or [
        (payload["name"], seed, original_epoch)
        for payload in continuation["payloads"]
        for seed in cfg["seeds"]
        for original_epoch in range(1, cfg["epochs"] + 1)
    ]
    cells = []
    for prefix_name, seed, original_epoch in selections:
        payload = payloads_by_name.get(prefix_name)
        if payload is None:
            raise AwsTrajectoryError(
                f"continuation cell references unknown prefix {prefix_name!r}"
            )
        raw = f"{treatment}-{prefix_name}-{seed}-e{original_epoch}"
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
        cell_id = f"{_safe_slug(raw)}-{suffix}"
        task_suffix = cell_id
        worker_run_dir = f"{campaign_id}-{cell_id}"
        cell_prefix = f"campaigns/{campaign_id}/cells/{cell_id}"
        model_last = payload["target_model"].split("/")[-1]
        task_name = bounded_inspect_task_name(
            f"continuation_{treatment}_{model_last}_{seed}_p{prefix_name}",
            task_suffix,
        )
        args = [
            f"--treatment={treatment}",
            f"--prefix-files={WORKER_PREFIX_PAYLOAD_PATH}",
            f"--seed-dir={cfg.get('seed_dir') or Path(cfg['seeds_path']).name}",
            f"--seeds={seed}",
            "--epochs=1",
            f"--harness={cfg['harness']}",
            f"--condition={cfg['condition']}",
            f"--judge={cfg['judge_resolved']}",
            f"--gate-model={cfg['gate_model']}",
            "--concurrency=1",
            "--sandbox-concurrency=1",
            f"--time-limit={cfg['time_limit']}",
            "--compute=local",
            "--skip-viewer",
        ]
        if cfg.get("pressure"):
            args.append(f"--pressure={cfg['pressure']}")
        if cfg.get("allow_incomplete_prefixes"):
            args.append("--allow-incomplete-prefixes")
        sandbox_compose = _sandbox_compose_for_target(
            cfg, target=payload["target"], target_model=payload["target_model"]
        )
        cells.append({
            "schema_version": AWS_SCHEMA_VERSION,
            "kind": "trajectory",
            "campaign_id": campaign_id,
            "cell_id": cell_id,
            "target": payload["target"],
            "target_model": payload["target_model"],
            "harness": cfg["harness"],
            "family": _seed_family(cfg),
            "sandbox_compose": sandbox_compose,
            "seed": seed,
            "original_epoch": original_epoch,
            "treatment": treatment,
            "prefix_name": prefix_name,
            "prefix_sha256": payload["sha256"],
            "prefix_payload_key": (
                f"campaigns/{campaign_id}/prefixes/"
                f"{prefix_name}-{payload['sha256'][:12]}.json"
            ),
            "prefix_payload_sha256": payload["file_sha256"],
            "prefix_payload_local": payload["local_path"],
            "pipeline_script": "exp_continuation_pipeline.py",
            "task_suffix": task_suffix,
            "task_name": task_name,
            "worker_run_dir": worker_run_dir,
            "bucket": bucket,
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "funding": cfg.get("aws_funding"),
            "hourly_price_usd": hourly_price,
            "root_volume_gb": ROOT_VOLUME_GB,
            "source_key": f"campaigns/{campaign_id}/source/source.tar.gz",
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "job_key": f"{cell_prefix}/job.json",
            "result_key": f"{cell_prefix}/result.tar.gz",
            "complete_key": f"{cell_prefix}/complete.json",
            "failure_key": f"{cell_prefix}/failure.json",
            "pipeline_args": args,
            "allowed_secret_names": list(cfg.get("worker_allowed_secret_names") or []),
            "status": "planned",
        })
    return cells


def build_multi_agent_cells(cfg: dict, *, campaign_id: str, source: dict,
                            bucket: str, hourly_price: float) -> list[dict]:
    """One VM per (activity log, new seed, epoch), mirroring continuations."""

    experiment = cfg["multi_agent"]
    treatment = experiment["treatment"]
    payloads_by_name = {item["name"]: item for item in experiment["payloads"]}
    selections = cfg.get("_cell_selections") or [
        (payload["name"], seed, original_epoch)
        for payload in experiment["payloads"]
        for seed in cfg["seeds"]
        for original_epoch in range(1, cfg["epochs"] + 1)
    ]
    cells = []
    for activity_name, seed, original_epoch in selections:
        payload = payloads_by_name.get(activity_name)
        if payload is None:
            raise AwsTrajectoryError(
                f"multi-agent cell references unknown activity log {activity_name!r}"
            )
        raw = f"{treatment}-{activity_name}-{seed}-e{original_epoch}"
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
        cell_id = f"{_safe_slug(raw)}-{suffix}"
        task_suffix = cell_id
        worker_run_dir = f"{campaign_id}-{cell_id}"
        cell_prefix = f"campaigns/{campaign_id}/cells/{cell_id}"
        model_last = payload["target_model"].split("/")[-1]
        task_name = bounded_inspect_task_name(
            f"multi_agent_{treatment}_{model_last}_{seed}_a{activity_name}",
            task_suffix,
        )
        args = [
            f"--treatment={treatment}",
            f"--activity-log-files={WORKER_ACTIVITY_LOG_PAYLOAD_PATH}",
            f"--seed-dir={cfg.get('seed_dir') or Path(cfg['seeds_path']).name}",
            f"--seeds={seed}",
            "--epochs=1",
            f"--harness={cfg['harness']}",
            f"--condition={cfg['condition']}",
            f"--judge={cfg['judge_resolved']}",
            f"--gate-model={cfg['gate_model']}",
            "--concurrency=1",
            "--sandbox-concurrency=1",
            f"--time-limit={cfg['time_limit']}",
            "--compute=local",
            "--skip-viewer",
        ]
        if cfg.get("pressure"):
            args.append(f"--pressure={cfg['pressure']}")
        sandbox_compose = _sandbox_compose_for_target(
            cfg, target=payload["target"], target_model=payload["target_model"]
        )
        payload_key = (
            f"campaigns/{campaign_id}/activity-logs/"
            f"{activity_name}-{payload['sha256'][:12]}.json"
        )
        cells.append({
            "schema_version": AWS_SCHEMA_VERSION,
            "kind": "trajectory",
            "campaign_id": campaign_id,
            "cell_id": cell_id,
            "target": payload["target"],
            "target_model": payload["target_model"],
            "harness": cfg["harness"],
            "family": _seed_family(cfg),
            "sandbox_compose": sandbox_compose,
            "seed": seed,
            "original_epoch": original_epoch,
            "treatment": treatment,
            "activity_log_name": activity_name,
            "activity_log_sha256": payload["sha256"],
            "activity_log_payload_key": payload_key,
            "activity_log_payload_sha256": payload["file_sha256"],
            "activity_log_payload_local": payload["local_path"],
            "pipeline_script": "exp_multi_agent_pipeline.py",
            "task_suffix": task_suffix,
            "task_name": task_name,
            "worker_run_dir": worker_run_dir,
            "bucket": bucket,
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "funding": cfg.get("aws_funding"),
            "hourly_price_usd": hourly_price,
            "root_volume_gb": ROOT_VOLUME_GB,
            "source_key": f"campaigns/{campaign_id}/source/source.tar.gz",
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "job_key": f"{cell_prefix}/job.json",
            "result_key": f"{cell_prefix}/result.tar.gz",
            "complete_key": f"{cell_prefix}/complete.json",
            "failure_key": f"{cell_prefix}/failure.json",
            "pipeline_args": args,
            "allowed_secret_names": list(
                cfg.get("worker_allowed_secret_names") or []
            ),
            "status": "planned",
        })
    return cells


def _upload_prefix_payloads(clients: dict, cells: list[dict]) -> None:
    """Upload each unique prefix payload once, verifying local bytes first."""

    uploaded: set[str] = set()
    for cell in cells:
        key = cell.get("prefix_payload_key")
        if not key or key in uploaded:
            continue
        local = Path(cell["prefix_payload_local"])
        if not local.is_file():
            raise AwsTrajectoryError(
                f"continuation prefix payload is missing locally: {local}"
            )
        digest = _sha256_file(local)
        if digest != cell["prefix_payload_sha256"]:
            raise AwsTrajectoryError(
                f"continuation prefix payload {local} changed since planning "
                f"(sha {digest[:12]} != recorded "
                f"{cell['prefix_payload_sha256'][:12]})"
            )
        clients["s3"].upload_file(
            str(local), cell["bucket"], key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        uploaded.add(key)


def _upload_activity_log_payloads(clients: dict, cells: list[dict]) -> None:
    """Upload each unique activity-log payload once with byte verification."""

    uploaded: set[str] = set()
    for cell in cells:
        key = cell.get("activity_log_payload_key")
        if not key or key in uploaded:
            continue
        local = Path(cell["activity_log_payload_local"])
        if not local.is_file():
            raise AwsTrajectoryError(
                f"activity-log payload is missing locally: {local}"
            )
        digest = _sha256_file(local)
        if digest != cell["activity_log_payload_sha256"]:
            raise AwsTrajectoryError(
                f"activity-log payload {local} changed since planning "
                f"({digest[:12]} != "
                f"{cell['activity_log_payload_sha256'][:12]})"
            )
        clients["s3"].upload_file(
            str(local), cell["bucket"], key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        uploaded.add(key)


def _campaign_id(cfg: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    nonce = uuid.uuid4().hex[:8]
    if cfg.get("continuation"):
        label = _safe_slug(cfg["continuation"]["treatment"])
        return f"continuation-aws-{label}-{cfg['epochs']}ep-{stamp}-{nonce}"
    if cfg.get("multi_agent"):
        label = _safe_slug(cfg["multi_agent"]["treatment"])
        return f"multi-agent-aws-{label}-{cfg['epochs']}ep-{stamp}-{nonce}"
    if cfg.get("prefix_only") is True:
        label = _safe_slug(cfg["name"])
        namespace = _safe_slug(
            str(cfg.get("prefix_campaign_namespace") or "ml-prefix")
        )
        return f"{namespace}-aws-{label}-{cfg['epochs']}ep-{stamp}-{nonce}"
    label = cfg["targets"][0] if len(cfg["targets"]) == 1 else f"{len(cfg['targets'])}targets"
    return (
        f"real-v2-aws-{_safe_slug(label)}-{cfg['condition']}-"
        f"{cfg['epochs']}ep-{stamp}-{nonce}"
    )


def _campaign_state_path(data_root: Path, campaign_id: str) -> Path:
    root = (
        REMOTE_CAMPAIGNS_ROOT
        if data_root == DATA_ROOT
        else data_root / "remote_campaigns"
    )
    return root / f"{campaign_id}.json"


def _campaign_logs_root(data_root: Path) -> Path:
    return LOGS_ROOT if data_root == DATA_ROOT else data_root / "logs"


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_campaign(state: dict, data_root: Path, clients: dict | None = None) -> None:
    state["updated_at"] = _utc_now()
    _write_json_atomic(_campaign_state_path(data_root, state["campaign_id"]), state)
    if clients:
        clients["s3"].put_object(
            Bucket=state["bucket"],
            Key=f"campaigns/{state['campaign_id']}/campaign.json",
            Body=json.dumps(state, sort_keys=True).encode(),
            ServerSideEncryption="AES256",
        )


def _campaign_s3_prefix(campaign_id: str) -> str:
    campaign_id = str(campaign_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign_id):
        raise AwsTrajectoryError(
            f"refusing unsafe S3 campaign identifier: {campaign_id!r}"
        )
    return f"campaigns/{campaign_id}/"


def delete_campaign_s3_objects(s3: Any, *, bucket: str, campaign_id: str) -> dict:
    """Delete one exact, validated campaign prefix and verify that it is empty."""
    prefix = _campaign_s3_prefix(campaign_id)
    objects: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents") or [])

    for offset in range(0, len(objects), 1000):
        batch = objects[offset:offset + 1000]
        response = s3.delete_objects(
            Bucket=bucket,
            Delete={
                "Objects": [{"Key": item["Key"]} for item in batch],
                "Quiet": True,
            },
        )
        errors = response.get("Errors") or []
        if errors:
            failed_keys = [str(error.get("Key")) for error in errors[:10]]
            raise AwsTrajectoryError(
                "S3 reported campaign cleanup errors for " + ", ".join(failed_keys)
            )

    remaining = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if remaining.get("Contents"):
        raise AwsTrajectoryError(
            f"S3 campaign prefix was not empty after cleanup: s3://{bucket}/{prefix}"
        )
    return {
        "prefix": prefix,
        "objects_deleted": len(objects),
        "bytes_deleted": sum(int(item.get("Size") or 0) for item in objects),
    }


def _verify_local_import_for_cleanup(state: dict, imported: Path) -> None:
    sidecar_path = imported / "remote_campaign.json"
    if not imported.is_dir() or not sidecar_path.is_file():
        raise AwsTrajectoryError("local campaign import is incomplete")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("campaign_id") != state["campaign_id"]:
        raise AwsTrajectoryError("local campaign sidecar does not match the campaign")

    source = state.get("source") or {}
    expected_source_sha = source.get("sha256")
    local_source = imported / "remote_source.tar.gz"
    if not expected_source_sha or not local_source.is_file():
        raise AwsTrajectoryError("local campaign import has no verified source archive")
    if _sha256_file(local_source) != expected_source_sha:
        raise AwsTrajectoryError("local campaign source checksum changed after import")


def _failed_s3_cleanup(state: dict, attempted_at: str, ex: Exception) -> dict:
    print(
        "  WARNING: local import is safe, but immediate S3 cleanup failed; "
        f"the {S3_RETENTION_DAYS}-day lifecycle remains the fallback: {ex}"
    )
    return {
        "status": "failed",
        "attempted_at": attempted_at,
        "prefix": _campaign_s3_prefix(state["campaign_id"]),
        "error": str(ex)[:500],
        "fallback_lifecycle_days": S3_RETENTION_DAYS,
    }


def _attempt_s3_cleanup(state: dict, clients: dict, attempted_at: str) -> dict:
    try:
        deleted = delete_campaign_s3_objects(
            clients["s3"], bucket=state["bucket"], campaign_id=state["campaign_id"]
        )
    except Exception as ex:
        return _failed_s3_cleanup(state, attempted_at, ex)
    cleanup = {
        "status": "deleted",
        "attempted_at": attempted_at,
        "completed_at": _utc_now(),
        **deleted,
    }
    print(
        "  deleted verified S3 handoff: "
        f"{cleanup['objects_deleted']} object(s), {cleanup['bytes_deleted']} bytes"
    )
    return cleanup


def _finalize_campaign_import(
    state: dict, data_root: Path, clients: dict, imported: Path
) -> dict:
    """Save the final state, then remove its now-redundant S3 handoff objects."""
    state["local_log_dir"] = str(imported)
    # Write the final recoverable state to both locations before deleting S3. This also
    # ensures a failed metadata upload leaves the campaign untouched in S3.
    _save_campaign(state, data_root, clients)
    attempted_at = _utc_now()
    try:
        _verify_local_import_for_cleanup(state, imported)
    except Exception as ex:
        cleanup = _failed_s3_cleanup(state, attempted_at, ex)
    else:
        cleanup = _attempt_s3_cleanup(state, clients, attempted_at)

    state["s3_cleanup"] = cleanup
    # Local-only save: uploading now would recreate campaign.json after deletion.
    _save_campaign(state, data_root)
    sidecar_path = imported / "remote_campaign.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["local_log_dir"] = str(imported)
    sidecar["s3_cleanup"] = cleanup
    _write_json_atomic(sidecar_path, sidecar)
    return state


def _worker_user_data(cell: dict) -> str:
    job_uri = shlex.quote(f"s3://{cell['bucket']}/{cell['job_key']}")
    source_uri = shlex.quote(f"s3://{cell['bucket']}/{cell['source_key']}")
    quoted_region = shlex.quote(str(cell["region"]))
    checksum_line = shlex.quote(
        f"{cell['source_sha256']}  /tmp/mats-source.tar.gz"
    )
    return f"""#!/bin/bash
set -uo pipefail
mkdir -p /var/lib/mats-worker /opt/supermats
exec > >(tee -a /var/lib/mats-worker/bootstrap.log) 2>&1
# The reusable AMI contains a builder-only timer guarded by the old builder's hardware
# UUID. Remove it from every real worker after that guard has safely no-op'd at boot.
systemctl disable --now mats-ami-builder-watchdog.timer 2>/dev/null || true
rm -f /etc/systemd/system/mats-ami-builder-watchdog.timer /etc/systemd/system/mats-ami-builder-watchdog.service /usr/local/sbin/mats-ami-builder-watchdog
systemctl daemon-reload
shutdown -h +{UNCONDITIONAL_TERMINATION_SECONDS // 60}
aws s3 cp {job_uri} /var/lib/mats-worker/job.json --region {quoted_region}
aws s3 cp {source_uri} /tmp/mats-source.tar.gz --region {quoted_region}
if ! echo {checksum_line} | sha256sum -c -; then
  exit_code=91
else
  rm -rf /opt/supermats/mats
  tar -xzf /tmp/mats-source.tar.gz -C /opt/supermats
  timeout --signal=TERM --kill-after=60 {FAILURE_PACKAGE_SECONDS} /opt/environments-venv/bin/python /opt/supermats/mats/environments/lib/exp_aws_trajectory.py --worker=/var/lib/mats-worker/job.json
  exit_code=$?
fi
if [ "$exit_code" -eq 91 ]; then
  /opt/environments-venv/bin/python /opt/supermats/mats/environments/lib/exp_aws_trajectory.py --watchdog-failure=/var/lib/mats-worker/job.json --failure-reason=source_checksum_failed || true
elif [ "$exit_code" -eq 124 ] || [ "$exit_code" -eq 137 ]; then
  /opt/environments-venv/bin/python /opt/supermats/mats/environments/lib/exp_aws_trajectory.py --watchdog-failure=/var/lib/mats-worker/job.json --failure-reason=worker_watchdog_expired || true
fi
shutdown -h now
exit "$exit_code"
"""


def _launch_cell(clients: dict, *, template_id: str, cell: dict) -> str:
    clients["s3"].put_object(
        Bucket=cell["bucket"], Key=cell["job_key"], Body=_json_bytes(cell),
        ServerSideEncryption="AES256",
    )
    run_args = {
        "LaunchTemplate": {
            "LaunchTemplateId": template_id,
            "Version": "$Default",
        },
        "MinCount": 1,
        "MaxCount": 1,
        "ClientToken": hashlib.sha256(
            f"{cell['campaign_id']}:{cell['cell_id']}".encode()
        ).hexdigest(),
        "UserData": _worker_user_data(cell),
        "TagSpecifications": [{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"{STACK_NAME}-{cell['cell_id']}"},
            {"Key": "Project", "Value": STACK_NAME},
            {"Key": "Campaign", "Value": cell["campaign_id"]},
            {"Key": "Cell", "Value": cell["cell_id"]},
        ]}],
    }
    for attempt in range(1, EC2_LAUNCH_MAX_ATTEMPTS + 1):
        try:
            response = clients["ec2"].run_instances(**run_args)
            break
        except Exception as error:
            code = _client_error_code(error)
            if code not in EC2_LAUNCH_THROTTLE_CODES or attempt >= EC2_LAUNCH_MAX_ATTEMPTS:
                raise
            backoff_cap = min(
                EC2_LAUNCH_BACKOFF_CAP_SECONDS,
                float(2 ** (attempt - 1)),
            )
            delay = random.uniform(0.5, backoff_cap)
            print(
                f"  {cell['cell_id']}: EC2 launch throttled ({code}); "
                f"retry {attempt + 1}/{EC2_LAUNCH_MAX_ATTEMPTS} in {delay:.1f}s"
            )
            time.sleep(delay)
    return response["Instances"][0]["InstanceId"]


def _s3_json_or_none(s3, bucket: str, key: str) -> dict | None:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as ex:
        if _client_error_code(ex) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return json.loads(response["Body"].read())


def _instance_records(ec2, instance_ids: list[str]) -> dict[str, dict]:
    if not instance_ids:
        return {}
    output = {}
    for start in range(0, len(instance_ids), 100):
        response = ec2.describe_instances(InstanceIds=instance_ids[start:start + 100])
        for reservation in response["Reservations"]:
            for instance in reservation["Instances"]:
                output[instance["InstanceId"]] = instance
    return output


def _nonterminal_campaign_instance_ids(ec2, campaign_id: str) -> set[str]:
    response = ec2.describe_instances(Filters=[
        {"Name": "tag:Campaign", "Values": [campaign_id]},
        {"Name": "instance-state-name", "Values": [
            "pending", "running", "stopping", "stopped", "shutting-down",
        ]},
    ])
    return {
        instance["InstanceId"]
        for reservation in response["Reservations"]
        for instance in reservation["Instances"]
    }


def _terminal_marker(clients: dict, cell: dict) -> tuple[str, dict] | None:
    complete = _s3_json_or_none(
        clients["s3"], cell["bucket"], cell["complete_key"]
    )
    if complete is not None:
        return "completed", complete
    failure = _s3_json_or_none(
        clients["s3"], cell["bucket"], cell["failure_key"]
    )
    if failure is not None:
        return "infrastructure_failure", failure
    return None


def _recover_campaign_instances(state: dict, clients: dict) -> bool:
    """Recover a launch if the laptop stopped after EC2 accepted it but before save."""
    response = clients["ec2"].describe_instances(Filters=[
        {"Name": "tag:Campaign", "Values": [state["campaign_id"]]},
        {"Name": "instance-state-name", "Values": [
            "pending", "running", "stopping", "stopped", "shutting-down", "terminated",
        ]},
    ])
    by_cell: dict[str, list[dict]] = {}
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            tags = _tags_dict(instance.get("Tags") or [])
            if tags.get("Cell"):
                by_cell.setdefault(tags["Cell"], []).append(instance)
    changed = False
    for cell in state["cells"]:
        if cell.get("instance_id"):
            continue
        matches = by_cell.get(cell["cell_id"]) or []
        if len(matches) > 1:
            raise AwsTrajectoryError(f"multiple EC2 instances found for {cell['cell_id']}")
        if matches:
            cell["instance_id"] = matches[0]["InstanceId"]
            cell["status"] = "launching"
            cell["recovered_launch"] = True
            changed = True
            print(f"  {cell['cell_id']}: recovered launch {cell['instance_id']}")
    return changed


def _monitor_campaign(state: dict, data_root: Path, clients: dict,
                      *, allow_launch: bool) -> dict:
    try:
        return _monitor_campaign_unchecked(
            state, data_root, clients, allow_launch=allow_launch
        )
    except Exception as error:
        recovery_message = _authentication_recovery_message(state, error)
        if recovery_message is None:
            raise
        # The normal save path writes locally before attempting S3. Repeat the local
        # write without AWS so even an expiry during remote save is safely resumable.
        _save_campaign(state, data_root)
        raise AwsTrajectoryError(recovery_message) from None


def _monitor_campaign_unchecked(state: dict, data_root: Path, clients: dict,
                                *, allow_launch: bool) -> dict:
    cells = state["cells"]
    concurrency = state["vm_concurrency"]
    print(f"campaign {state['campaign_id']}: {len(cells)} cells, max {concurrency} VMs")
    if _recover_campaign_instances(state, clients):
        _save_campaign(state, data_root, clients)
    while True:
        active = [
            cell for cell in cells
            if cell["status"] in {"launching", "running", "finishing"}
        ]
        records = _instance_records(
            clients["ec2"], [cell["instance_id"] for cell in active]
        )
        missing_active_ids = {
            cell["instance_id"]
            for cell in active
            if cell["instance_id"] not in records
        }
        nonterminal_ids = (
            _nonterminal_campaign_instance_ids(
                clients["ec2"], state["campaign_id"]
            )
            if missing_active_ids
            else set()
        )
        changed = False
        for cell in active:
            marker = _terminal_marker(clients, cell)
            record = records.get(cell["instance_id"], {})
            instance_state = (record.get("State") or {}).get("Name")
            if record.get("LaunchTime") and not cell.get("launched_at"):
                launched = record["LaunchTime"]
                cell["launched_at"] = (
                    launched.isoformat() if hasattr(launched, "isoformat") else str(launched)
                )
                changed = True
            if marker and not cell.get("terminal"):
                status, terminal = marker
                instance_is_terminal = (
                    instance_state == "terminated"
                    or (
                        cell["instance_id"] in missing_active_ids
                        and cell["instance_id"] not in nonterminal_ids
                    )
                )
                cell["terminal"] = terminal
                cell["completed_at"] = terminal.get("completed_at") or _utc_now()
                cell["terminal_status"] = status
                cell["status"] = status if instance_is_terminal else "finishing"
                if instance_is_terminal:
                    cell.pop("terminal_status", None)
                    cell["instance_state"] = "terminated"
                    cell["terminated_at"] = _utc_now()
                    if instance_state != "terminated":
                        cell["terminated_at_is_upper_bound"] = True
                        cell["termination_resolution"] = (
                            "absent_from_all_nonterminal_campaign_instances"
                        )
                changed = True
                if instance_is_terminal:
                    print(
                        f"  {cell['cell_id']}: {status}; "
                        "VM already terminated or no longer present"
                    )
                else:
                    print(f"  {cell['cell_id']}: result uploaded; waiting for VM termination")
            elif cell["status"] == "finishing" and instance_state == "terminated":
                cell["status"] = cell.pop("terminal_status")
                cell["instance_state"] = instance_state
                cell["terminated_at"] = _utc_now()
                changed = True
                print(f"  {cell['cell_id']}: {cell['status']}; VM terminated")
            elif (
                cell["status"] == "finishing"
                and cell.get("terminal")
                and cell["instance_id"] in missing_active_ids
                and cell["instance_id"] not in nonterminal_ids
            ):
                observed_at = _utc_now()
                cell["status"] = cell.pop("terminal_status")
                cell["instance_state"] = "terminated"
                cell["terminated_at"] = observed_at
                cell["terminated_at_is_upper_bound"] = True
                cell["termination_resolution"] = (
                    "absent_from_all_nonterminal_campaign_instances"
                )
                changed = True
                print(
                    f"  {cell['cell_id']}: {cell['status']}; "
                    "VM no longer present in any non-terminal state"
                )
            elif (
                cell["instance_id"] in missing_active_ids
                and cell["instance_id"] not in nonterminal_ids
            ):
                observed_at = _utc_now()
                cell["status"] = "infrastructure_failure"
                cell["instance_state"] = "terminated"
                cell["completed_at"] = observed_at
                cell["terminated_at"] = observed_at
                cell["terminated_at_is_upper_bound"] = True
                cell["termination_resolution"] = (
                    "absent_from_all_nonterminal_campaign_instances"
                )
                cell["terminal"] = {
                    "reason": (
                        "instance no longer present and no completion/failure marker exists"
                    ),
                }
                changed = True
                print(
                    f"  {cell['cell_id']}: infrastructure failure; "
                    "VM no longer present and no marker exists"
                )
            elif instance_state == "terminated":
                cell["status"] = "infrastructure_failure"
                cell["instance_state"] = instance_state
                cell["completed_at"] = _utc_now()
                cell["terminated_at"] = _utc_now()
                cell["terminal"] = {
                    "reason": "instance terminated without a completion/failure marker",
                    "instance_state": instance_state,
                }
                changed = True
                print(f"  {cell['cell_id']}: infrastructure failure (no marker)")
            elif instance_state == "shutting-down" and not cell.get("terminal"):
                cell["status"] = "finishing"
                cell["terminal_status"] = "infrastructure_failure"
                cell["completed_at"] = _utc_now()
                cell["terminal"] = {
                    "reason": "instance began terminating without a completion/failure marker",
                    "instance_state": instance_state,
                }
                changed = True
                print(f"  {cell['cell_id']}: terminating; waiting before freeing VM slot")
            elif instance_state and instance_state != cell.get("instance_state"):
                cell["instance_state"] = instance_state
                cell["status"] = "running" if instance_state == "running" else cell["status"]
                changed = True
            launched_at = cell.get("launched_at") or cell.get("launcher_started_at")
            if cell["status"] in {"launching", "running", "finishing"} and launched_at:
                try:
                    launched_dt = datetime.fromisoformat(
                        str(launched_at).replace("Z", "+00:00")
                    )
                    overdue = (
                        datetime.now(timezone.utc) - launched_dt
                    ).total_seconds() > UNCONDITIONAL_TERMINATION_SECONDS + 10 * 60
                except Exception:
                    overdue = False
                if overdue and not cell.get("termination_requested_at"):
                    clients["ec2"].terminate_instances(InstanceIds=[cell["instance_id"]])
                    terminal_status = cell.pop("terminal_status", None)
                    had_uploaded_marker = terminal_status is not None
                    terminal_status = terminal_status or "infrastructure_failure"
                    cell["terminal_status"] = terminal_status
                    cell["status"] = "finishing"
                    cell["completed_at"] = _utc_now()
                    if had_uploaded_marker:
                        cell["forced_termination_after_upload"] = True
                    if not cell.get("terminal"):
                        cell["terminal"] = {
                            "reason": "controller terminated VM after the hard watchdog grace",
                        }
                    cell["termination_requested_at"] = _utc_now()
                    changed = True

        active_count = sum(
            cell["status"] in {"launching", "running", "finishing"} for cell in cells
        )
        if allow_launch:
            for cell in cells:
                if active_count >= concurrency:
                    break
                if cell["status"] != "planned":
                    continue
                instance_id = _launch_cell(
                    clients, template_id=state["launch_template_id"], cell=cell
                )
                cell.update({
                    "status": "launching",
                    "instance_id": instance_id,
                    "launcher_started_at": _utc_now(),
                })
                active_count += 1
                changed = True
                print(f"  {cell['cell_id']}: launched {instance_id}")
                _save_campaign(state, data_root, clients)

        if changed:
            _save_campaign(state, data_root, clients)
        if all(cell["status"] not in {"planned", "launching", "running", "finishing"}
               for cell in cells):
            break
        if not allow_launch and not active_count:
            for cell in cells:
                if cell["status"] == "planned":
                    cell["status"] = "not_launched"
                    cell["terminal"] = {
                        "reason": "launcher stopped before this cell was launched; resume never launches"
                    }
            _save_campaign(state, data_root, clients)
            break
        time.sleep(15)
    return state


def _secure_extract_result(archive_path: Path, destination: Path) -> dict:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise AwsTrajectoryError(f"unsafe result archive member: {member.name}")
        manifest_member = archive.getmember("manifest.json")
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise AwsTrajectoryError("result archive manifest is unreadable")
        manifest = json.loads(manifest_stream.read())
        archive.extractall(destination, filter="data")
    entries = manifest.get("files") or []
    expected_paths: set[str] = set()
    for entry in entries:
        relative = PurePosixPath(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise AwsTrajectoryError(f"unsafe result manifest path: {entry['path']}")
        normalized = relative.as_posix()
        if normalized in expected_paths:
            raise AwsTrajectoryError(f"duplicate result manifest path: {normalized}")
        expected_paths.add(normalized)
        path = destination / normalized
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise AwsTrajectoryError(f"result checksum mismatch: {entry['path']}")
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.relative_to(destination).as_posix() != "manifest.json"
    }
    if actual_paths != expected_paths:
        raise AwsTrajectoryError(
            "result manifest file set mismatch: "
            f"unlisted={sorted(actual_paths - expected_paths)}, "
            f"missing={sorted(expected_paths - actual_paths)}"
        )
    return manifest


def _vm_cost(cell: dict) -> dict:
    terminal = cell.get("terminal") or {}
    started = cell.get("launched_at") or cell.get("launcher_started_at")
    finished = (
        cell.get("terminated_at")
        or terminal.get("completed_at")
        or cell.get("completed_at")
    )
    seconds = None
    try:
        start_dt = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        finish_dt = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
        seconds = max(60.0, (finish_dt - start_dt).total_seconds())
    except Exception:
        pass
    price = float(cell.get("hourly_price_usd") or 0.0)
    return {
        "provider": "aws",
        "region": cell.get("region"),
        "instance_type": cell.get("instance_type"),
        "instance_id": cell.get("instance_id"),
        "funding": cell.get("funding"),
        "campaign_id": cell.get("campaign_id"),
        "cell_id": cell.get("cell_id"),
        "original_epoch": cell.get("original_epoch"),
        "launched_at": started,
        "completed_at": finished,
        "billed_seconds_estimate": seconds,
        "hourly_price_usd": price,
        "estimated_vm_cost_usd": None if seconds is None else seconds / 3600 * price,
        "source_sha256": cell.get("source_sha256"),
        "source_bytes": cell.get("source_bytes"),
        "result_bundle_bytes": terminal.get("result_bytes"),
        "s3_cost_excluded": True,
        "ebs_cost_excluded": True,
        "public_ipv4_cost_excluded": True,
        "internet_data_transfer_cost_excluded": True,
        "shared_runtime_cost_excluded": True,
        "root_volume_gb": int(
            cell.get("root_volume_gb") or LEGACY_ROOT_VOLUME_GB
        ),
    }


def import_campaign_results(state: dict, data_root: Path, clients: dict) -> Path:
    destination = _campaign_logs_root(data_root) / state["campaign_id"]
    if destination.exists():
        sidecar = destination / "remote_campaign.json"
        if sidecar.is_file() and json.loads(sidecar.read_text()).get("campaign_id") == state[
            "campaign_id"
        ]:
            print(f"  local campaign already imported: {destination}")
            return destination
        raise AwsTrajectoryError(f"refusing to overwrite existing log directory: {destination}")
    completed = [cell for cell in state["cells"] if cell["status"] == "completed"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{state['campaign_id']}.", dir=destination.parent))
    try:
        source_record = state.get("source") or {}
        if source_record.get("sha256"):
            source_archive = staging / "remote_source.tar.gz"
            clients["s3"].download_file(
                state["bucket"],
                f"campaigns/{state['campaign_id']}/source/source.tar.gz",
                str(source_archive),
            )
            if _sha256_file(source_archive) != source_record["sha256"]:
                raise AwsTrajectoryError("downloaded campaign source checksum mismatch")
        remote_cells = staging / "remote_cells"
        remote_cells.mkdir()
        task_compute: dict[str, dict] = {}
        integrity_records: list[dict] = []
        for cell in completed:
            terminal = cell.get("terminal") or {}
            archive_path = remote_cells / f"{cell['cell_id']}.tar.gz"
            clients["s3"].download_file(
                state["bucket"], cell["result_key"], str(archive_path)
            )
            expected = terminal.get("result_sha256")
            if expected and _sha256_file(archive_path) != expected:
                raise AwsTrajectoryError(f"S3 result checksum mismatch: {cell['cell_id']}")
            extract_dir = remote_cells / cell["cell_id"]
            extract_dir.mkdir()
            manifest = _secure_extract_result(archive_path, extract_dir)
            archive_path.unlink()
            run_root = extract_dir / "payload" / "run"
            if not run_root.is_dir():
                raise AwsTrajectoryError(f"result has no run directory: {cell['cell_id']}")
            for source in run_root.iterdir():
                if source.name == "pipeline_integrity.json" and source.is_file():
                    try:
                        integrity_payload = json.loads(source.read_text())
                    except json.JSONDecodeError as ex:
                        raise AwsTrajectoryError(
                            f"invalid pipeline integrity sidecar: {cell['cell_id']}: {ex}"
                        ) from ex
                    records = integrity_payload.get("records")
                    if not isinstance(records, list):
                        raise AwsTrajectoryError(
                            f"pipeline integrity sidecar has no records: {cell['cell_id']}"
                        )
                    integrity_records.extend(
                        record for record in records if isinstance(record, dict)
                    )
                if source.suffix == ".eval":
                    target = staging / source.name
                elif source.name == "real_artifacts":
                    target = staging / "real_artifacts"
                else:
                    target = extract_dir / "run_sidecars" / source.name
                    target.parent.mkdir(exist_ok=True)
                if target.exists() and source.is_dir():
                    for child in source.iterdir():
                        child_target = target / child.name
                        if child_target.exists():
                            raise AwsTrajectoryError(f"result path collision: {child_target}")
                        shutil.move(str(child), child_target)
                    source.rmdir()
                elif target.exists():
                    raise AwsTrajectoryError(f"result path collision: {target}")
                else:
                    shutil.move(str(source), target)
            task_name = terminal.get("task_name")
            if task_name:
                task_compute[str(task_name)] = _vm_cost(cell)
            cell["result_manifest"] = manifest
        # Infrastructure-failure bundles are preserved for diagnosis but never merged
        # as ordinary .eval files: they may have been captured while Inspect was writing.
        for cell in state["cells"]:
            terminal = cell.get("terminal") or {}
            if cell["status"] != "infrastructure_failure" or not terminal.get(
                "result_sha256"
            ):
                continue
            archive_path = remote_cells / f"{cell['cell_id']}-failure.tar.gz"
            clients["s3"].download_file(
                state["bucket"], cell["result_key"], str(archive_path)
            )
            if _sha256_file(archive_path) != terminal["result_sha256"]:
                raise AwsTrajectoryError(
                    f"failure-bundle checksum mismatch: {cell['cell_id']}"
                )
        sidecar_state = {
            **{key: value for key, value in state.items() if key != "clients"},
            "task_compute": task_compute,
            "imported_at": _utc_now(),
            "storage_cost_note": (
                "S3 request/storage, per-worker EBS, public IPv4, internet data "
                "transfer, and shared AMI builder/snapshot charges are excluded from "
                "trajectory cost totals; source/result byte counts, root-volume size, "
                "and exclusion flags are recorded."
            ),
        }
        (staging / "remote_campaign.json").write_text(
            json.dumps(sidecar_state, indent=2, sort_keys=True) + "\n"
        )
        if integrity_records:
            (staging / "pipeline_integrity.json").write_text(
                json.dumps({
                    "schema_version": "environment-pipeline-integrity-v2",
                    "records": integrity_records,
                }, indent=2, sort_keys=True) + "\n"
            )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"  imported {len(completed)} verified result(s) -> {destination}")
    return destination


def run_campaign(cfg: dict, environments_root: Path, data_root: Path,
                 *, dry_run: bool = False) -> dict:
    cfg = {
        **cfg,
        "family": _seed_family(cfg),
        "pressure": cfg.get("pressure"),
    }
    time_limit = cfg.get("time_limit")
    if not isinstance(time_limit, int) or time_limit <= 0:
        raise AwsTrajectoryError(
            "AWS campaigns require a finite, positive --time-limit so the worker "
            "watchdogs cannot silently truncate a trajectory"
        )
    if time_limit > AGENT_TIME_LIMIT_SECONDS:
        raise AwsTrajectoryError(
            f"AWS campaigns currently support --time-limit up to "
            f"{AGENT_TIME_LIMIT_SECONDS}s; use --compute=local for a longer run"
        )
    continuation = cfg.get("continuation")
    multi_agent = cfg.get("multi_agent")
    prefix_only = cfg.get("prefix_only") is True
    if sum(bool(mode) for mode in (continuation, multi_agent, prefix_only)) > 1:
        raise AwsTrajectoryError(
            "a campaign cannot combine continuation, multi-agent, and prefix-only "
            "generation modes"
        )
    selected_cells = cfg.get("_cell_selections")
    if selected_cells:
        planned_cells = len(selected_cells)
    elif continuation:
        planned_cells = (
            len(continuation["payloads"]) * len(cfg["seeds"]) * int(cfg["epochs"])
        )
    elif multi_agent:
        planned_cells = (
            len(multi_agent["payloads"]) * len(cfg["seeds"]) * int(cfg["epochs"])
        )
    else:
        planned_cells = len(cfg["targets"]) * len(cfg["seeds"]) * int(cfg["epochs"])
    if planned_cells < 1:
        raise AwsTrajectoryError("AWS campaign has no trajectory cells to run")
    # Quota and scheduling should reflect the work that can actually run. In
    # particular, an n=1 test must not require the default 250-worker quota.
    cfg = {
        **cfg,
        "vm_concurrency": min(int(cfg["vm_concurrency"]), planned_cells),
    }
    if continuation or multi_agent:
        conditioned_payloads = (continuation or multi_agent)["payloads"]
        target_specs = [
            (payload["target"], payload["target_model"], payload.get("reasoning"))
            for payload in conditioned_payloads
        ]
    else:
        target_specs = [
            (target, model, cfg.get("reasoning"))
            for target, model in zip(
                cfg["targets"], cfg["target_models"], strict=True
            )
        ]
    # A target can occur in multiple conditioned payloads. Preserve one copy while
    # rejecting inconsistent metadata before any preflight or spend.
    unique_target_specs: dict[str, tuple[str, bool | None]] = {}
    for target, model, reasoning in target_specs:
        prior = unique_target_specs.get(target)
        current = (model, reasoning)
        if prior is not None and prior != current:
            raise AwsTrajectoryError(
                f"target {target!r} has inconsistent model/reasoning metadata across "
                "conditioned payloads"
            )
        unique_target_specs[target] = current

    api_target_models = [model for model, _reasoning in unique_target_specs.values()]
    subscription_required: set[str] = set()
    subscription_alternatives: list[tuple[str, ...]] = []
    if cfg.get("harness") == "subscription":
        api_target_models = []
        for target, (slug, target_reasoning) in unique_target_specs.items():
            scaffold = production_scaffold_for_target(target, slug)
            if scaffold == "opencode":
                if opencode_go_model_spec(target, slug) is not None:
                    if not target_reasoning:
                        raise AwsTrajectoryError(
                            "OpenCode Go subscription runs currently require "
                            "--reasoning=yes; refusing to silently change reasoning "
                            "behavior"
                        )
                    subscription_required.add(OPENCODE_GO_API_KEY_ENV)
                else:
                    api_target_models.append(slug)
            elif scaffold == "claude_code":
                alternatives = (
                    "CLAUDE_CODE_OAUTH_TOKEN",
                    CLAUDE_AUTH_B64_ENV,
                )
                if alternatives not in subscription_alternatives:
                    subscription_alternatives.append(alternatives)
            elif scaffold == "codex":
                alternatives = (
                    "CODEX_ACCESS_TOKEN",
                    "CODEX_SUBSCRIPTION_AUTH_JSON_GZIP_B64",
                    "CODEX_SUBSCRIPTION_AUTH_JSON_B64",
                )
                if alternatives not in subscription_alternatives:
                    subscription_alternatives.append(alternatives)
    with tempfile.TemporaryDirectory(prefix="mats-aws-source-") as tmp:
        source = build_source_bundle(environments_root, Path(tmp))
        model_slugs = list(api_target_models)
        if not prefix_only:
            model_slugs.extend([cfg["judge_resolved"], cfg["gate_model"]])
        preflight = preflight_aws(
            cfg, environments_root,
            model_slugs=model_slugs,
            required_secrets=subscription_required,
            required_secret_alternatives=subscription_alternatives,
        )
        campaign_id = cfg.get("campaign_id") or _campaign_id(cfg)
        worker_cfg = {
            **cfg,
            "aws_funding": preflight["funding"],
            "worker_allowed_secret_names": preflight["stored_secret_names"],
        }
        if continuation:
            cell_builder = build_continuation_cells
        elif multi_agent:
            cell_builder = build_multi_agent_cells
        elif prefix_only:
            cell_builder = build_prefix_generation_cells
        else:
            cell_builder = build_cells
        cells = cell_builder(
            worker_cfg, campaign_id=campaign_id, source=source,
            bucket=preflight["bucket"], hourly_price=preflight["hourly_price_usd"],
        )
        worst_compute = len(cells) * (UNCONDITIONAL_TERMINATION_SECONDS / 3600) * preflight[
            "hourly_price_usd"
        ]
        summary = {
            "campaign_id": campaign_id,
            "cells": len(cells),
            "family": _seed_family(cfg),
            "sandbox_compose_files": sorted({
                cell["sandbox_compose"] for cell in cells
            }),
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "source_files": source["files"],
            "ami": preflight["ami"],
            "runtime_hash": preflight["runtime_hash"],
            "region": cfg["aws_region"],
            "funding": preflight["funding"],
            "instance_type": cfg["aws_instance_type"],
            "hourly_price_usd": preflight["hourly_price_usd"],
            "root_volume_gb": preflight["root_volume_gb"],
            "vm_concurrency": cfg["vm_concurrency"],
            "quota_vcpus": preflight["quota_vcpus"],
            "required_vcpus": preflight["required_vcpus"],
            "worst_case_compute_usd": worst_compute,
            "cost_exclusions": [
                "S3 requests and storage",
                "per-worker EBS",
                "public IPv4",
                "internet data transfer",
                "shared AMI builder and snapshot storage",
            ],
        }
        print("AWS campaign preflight:\n" + json.dumps(summary, indent=2, sort_keys=True))
        if dry_run:
            return {"dry_run": True, **summary, "cell_specs": cells}
        preflight["clients"]["s3"].upload_file(
            source["path"], preflight["bucket"],
            f"campaigns/{campaign_id}/source/source.tar.gz",
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        _upload_prefix_payloads(preflight["clients"], cells)
        _upload_activity_log_payloads(preflight["clients"], cells)
        state = {
            "schema_version": AWS_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "retry_parent": cfg.get("retry_parent"),
            "created_at": _utc_now(),
            "bucket": preflight["bucket"],
            "funding": preflight["funding"],
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "hourly_price_usd": preflight["hourly_price_usd"],
            "root_volume_gb": preflight["root_volume_gb"],
            "vm_concurrency": cfg["vm_concurrency"],
            "launch_template_id": preflight["launch_template_id"],
            "source": {key: source[key] for key in ("sha256", "bytes", "files")},
            "runtime_hash": preflight["runtime_hash"],
            "pipeline_config": {
                key: cfg[key]
                for key in (
                    "targets", "seeds", "seeds_path", "family", "epochs", "reasoning",
                    "harness", "condition", "judge_resolved", "gate_model", "time_limit",
                    "target_models", "pressure", "aws_secret_env",
                )
            } | (
                {
                    "continuation": continuation,
                    "seed_dir": cfg.get("seed_dir"),
                    "allow_incomplete_prefixes": bool(
                        cfg.get("allow_incomplete_prefixes")
                    ),
                }
                if continuation
                else (
                    {
                        "multi_agent": multi_agent,
                        "seed_dir": cfg.get("seed_dir"),
                    }
                    if multi_agent
                    else
                    {
                        "prefix_only": True,
                        "name": cfg["name"],
                        "seed_dir": cfg.get("seed_dir"),
                        "prefix_pipeline_script": cfg.get(
                            "prefix_pipeline_script", "prefixes/exp_ml_prefix.py"
                        ),
                        "prefix_task_namespace": cfg.get(
                            "prefix_task_namespace", "ml_prefix_only"
                        ),
                        "prefix_campaign_namespace": cfg.get(
                            "prefix_campaign_namespace", "ml-prefix"
                        ),
                    }
                    if prefix_only
                    else {}
                )
            ),
            "cells": cells,
        }
        _save_campaign(state, data_root, preflight["clients"])
        state = _monitor_campaign(
            state, data_root, preflight["clients"], allow_launch=True
        )
        imported = import_campaign_results(state, data_root, preflight["clients"])
        return _finalize_campaign_import(
            state, data_root, preflight["clients"], imported
        )


def load_campaign(campaign_id: str, data_root: Path, clients: dict,
                  bucket: str) -> dict:
    local = _campaign_state_path(data_root, campaign_id)
    if local.is_file():
        return json.loads(local.read_text())
    remote = _s3_json_or_none(clients["s3"], bucket, f"campaigns/{campaign_id}/campaign.json")
    if remote is None:
        raise AwsTrajectoryError(f"campaign not found locally or in S3: {campaign_id}")
    _write_json_atomic(local, remote)
    return remote


def _campaign_connection(cfg: dict, data_root: Path,
                         campaign_id: str) -> tuple[dict, str, dict | None]:
    """Use stored campaign location unless the user explicitly overrides it."""
    local_path = _campaign_state_path(data_root, campaign_id)
    local_state = json.loads(local_path.read_text()) if local_path.is_file() else None
    region = cfg["aws_region"]
    if local_state and not cfg.get("aws_region_explicit"):
        region = local_state.get("region") or region
    clients = aws_clients(region)
    if cfg.get("aws_bucket_explicit"):
        bucket = cfg["aws_bucket"]
    elif local_state:
        bucket = local_state.get("bucket")
    else:
        bucket = cfg.get("aws_bucket")
    bucket = bucket or default_bucket_name(account_id(clients), region)
    return clients, str(bucket), local_state


def resume_campaign(cfg: dict, environments_root: Path, data_root: Path,
                    campaign_id: str) -> dict:
    clients, bucket, local_state = _campaign_connection(cfg, data_root, campaign_id)
    state = local_state or load_campaign(campaign_id, data_root, clients, bucket)
    state = _monitor_campaign(state, data_root, clients, allow_launch=False)
    imported = import_campaign_results(state, data_root, clients)
    return _finalize_campaign_import(state, data_root, clients, imported)


def retry_failed(cfg: dict, environments_root: Path, data_root: Path,
                 campaign_id: str, *, dry_run: bool = False) -> dict:
    clients, bucket, local_state = _campaign_connection(cfg, data_root, campaign_id)
    old = local_state or load_campaign(campaign_id, data_root, clients, bucket)
    original_cfg = old.get("pipeline_config")
    if not isinstance(original_cfg, dict):
        raise AwsTrajectoryError(
            "campaign predates stored retry configuration; retry cannot safely guess"
        )
    prefix_only_retry = original_cfg.get("prefix_only") is True
    include_pipeline_failures = bool(cfg.get("retry_pipeline_failures"))

    def retryable(cell: dict) -> bool:
        if cell.get("status") in {"infrastructure_failure", "not_launched"}:
            return True
        if (
            not (prefix_only_retry or include_pipeline_failures)
            or cell.get("status") != "completed"
        ):
            return False
        exit_code = (cell.get("terminal") or {}).get("pipeline_exit_code")
        try:
            return int(exit_code) != 0
        except (TypeError, ValueError):
            return True

    failed = [
        cell for cell in old["cells"]
        if retryable(cell)
    ]
    if not failed:
        raise AwsTrajectoryError(
            f"campaign {campaign_id} has no retryable infrastructure failures"
            + (
                " or failed prefix-generation cells"
                if prefix_only_retry
                else " or completed pipeline failures"
                if include_pipeline_failures
                else ""
            )
        )
    original_harness = str(original_cfg.get("harness") or "simple")
    requested_harness = cfg.get("harness")
    if requested_harness != original_harness:
        raise AwsTrajectoryError(
            "--harness for --retry-failed must match the original campaign "
            f"({original_harness}); got {requested_harness!r}"
        )
    # Reconstruct the exact failed cell selection. This creates a new, visibly linked
    # campaign and never overwrites or cross-products the original evidence.
    retry_cfg = {**original_cfg, **cfg, "harness": original_harness}
    for config_key, state_key in (
        ("aws_region", "region"),
        ("aws_instance_type", "instance_type"),
        ("vm_concurrency", "vm_concurrency"),
        ("aws_bucket", "bucket"),
    ):
        if not cfg.get(f"{config_key}_explicit") and old.get(state_key) is not None:
            retry_cfg[config_key] = old[state_key]
    continuation_retry = bool(original_cfg.get("continuation"))
    multi_agent_retry = bool(original_cfg.get("multi_agent"))
    conditioned_retry = continuation_retry or multi_agent_retry
    retry_targets = list(dict.fromkeys(cell["target"] for cell in failed))
    if not conditioned_retry:
        try:
            original_target_models = dict(zip(
                original_cfg["targets"],
                original_cfg["target_models"],
                strict=True,
            ))
            retry_target_models = [
                original_target_models[target] for target in retry_targets
            ]
        except (KeyError, TypeError, ValueError) as ex:
            raise AwsTrajectoryError(
                "campaign has an invalid stored target/model mapping; retry cannot "
                "safely guess"
            ) from ex
    retry_cfg.update({
        "targets": retry_targets,
        "seeds": list(dict.fromkeys(cell["seed"] for cell in failed)),
        # Conditioned cells are keyed by their payload, plain cells by agent.
        "_cell_selections": [
            (
                (
                    cell["prefix_name"]
                    if continuation_retry
                    else cell["activity_log_name"]
                    if multi_agent_retry
                    else cell["target"]
                ),
                cell["seed"],
                cell["original_epoch"],
            )
            for cell in failed
        ],
        "campaign_id": (
            f"{campaign_id}-retry-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        ),
        "retry_parent": campaign_id,
    })
    if not conditioned_retry:
        retry_cfg["target_models"] = retry_target_models
    result = run_campaign(retry_cfg, environments_root, data_root, dry_run=dry_run)
    result["retry_parent"] = campaign_id
    return result


def campaign_ok(state: dict) -> bool:
    return bool(state.get("cells")) and all(
        cell.get("status") == "completed"
        and int((cell.get("terminal") or {}).get("pipeline_exit_code", 1)) == 0
        for cell in state["cells"]
    )


def smoke_test(cfg: dict, environments_root: Path, data_root: Path,
               *, dry_run: bool = False) -> dict:
    """Launch one no-LLM VM and verify every sandbox + the S3 result round trip."""
    with tempfile.TemporaryDirectory(prefix="mats-aws-smoke-source-") as tmp:
        source = build_source_bundle(environments_root, Path(tmp))
        try:
            sandbox_compose_paths = discover_sandbox_compose_files(environments_root)
        except ValueError as error:
            raise AwsTrajectoryError(str(error)) from None
        smoke_cfg = {**cfg, "vm_concurrency": 1}
        preflight = preflight_aws(smoke_cfg, environments_root, model_slugs=[])
        campaign_id = (
            f"aws-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        prefix = f"campaigns/{campaign_id}/cells/smoke"
        cell = {
            "schema_version": AWS_SCHEMA_VERSION,
            "kind": "smoke",
            "campaign_id": campaign_id,
            "cell_id": "smoke",
            "task_name": None,
            "worker_run_dir": f"{campaign_id}-smoke",
            "bucket": preflight["bucket"],
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "funding": preflight["funding"],
            "hourly_price_usd": preflight["hourly_price_usd"],
            "root_volume_gb": preflight["root_volume_gb"],
            "source_key": f"campaigns/{campaign_id}/source/source.tar.gz",
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "sandbox_compose_paths": sandbox_compose_paths,
            "job_key": f"{prefix}/job.json",
            "result_key": f"{prefix}/result.tar.gz",
            "complete_key": f"{prefix}/complete.json",
            "failure_key": f"{prefix}/failure.json",
            "status": "planned",
        }
        summary = {
            "campaign_id": campaign_id,
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "sandbox_compose_files": sandbox_compose_paths,
            "ami": preflight["ami"],
            "region": cfg["aws_region"],
            "funding": preflight["funding"],
            "instance_type": cfg["aws_instance_type"],
            "hourly_price_usd": preflight["hourly_price_usd"],
            "root_volume_gb": preflight["root_volume_gb"],
            "quota_vcpus": preflight["quota_vcpus"],
            "required_vcpus": preflight["required_vcpus"],
            "calls_models": False,
        }
        print("AWS no-LLM smoke preflight:\n" + json.dumps(summary, indent=2, sort_keys=True))
        if dry_run:
            return {"dry_run": True, **summary}
        preflight["clients"]["s3"].upload_file(
            source["path"], preflight["bucket"], cell["source_key"],
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        state = {
            "schema_version": AWS_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "created_at": _utc_now(),
            "bucket": preflight["bucket"],
            "funding": preflight["funding"],
            "region": cfg["aws_region"],
            "instance_type": cfg["aws_instance_type"],
            "hourly_price_usd": preflight["hourly_price_usd"],
            "root_volume_gb": preflight["root_volume_gb"],
            "vm_concurrency": 1,
            "launch_template_id": preflight["launch_template_id"],
            "source": {key: source[key] for key in ("sha256", "bytes", "files")},
            "runtime_hash": preflight["runtime_hash"],
            "smoke_test": True,
            "cells": [cell],
        }
        _save_campaign(state, data_root, preflight["clients"])
        state = _monitor_campaign(state, data_root, preflight["clients"], allow_launch=True)
        if not campaign_ok(state):
            return state
        smoke_dir = data_root / "aws_smoke" / campaign_id
        smoke_dir.mkdir(parents=True, exist_ok=False)
        archive = smoke_dir / "result.tar.gz"
        preflight["clients"]["s3"].download_file(
            preflight["bucket"], cell["result_key"], str(archive)
        )
        expected = (cell.get("terminal") or {}).get("result_sha256")
        if expected and _sha256_file(archive) != expected:
            raise AwsTrajectoryError("smoke result checksum mismatch")
        extracted = smoke_dir / "verified"
        extracted.mkdir()
        manifest = _secure_extract_result(archive, extracted)
        record = json.loads((extracted / "payload/run/smoke.json").read_text())
        if (
            not record.get("docker_info_ok")
            or not record.get("compose_config_ok")
            or not record.get("bubblewrap_ok")
        ):
            raise AwsTrajectoryError(f"smoke worker checks failed: {record}")
        local_source = smoke_dir / "remote_source.tar.gz"
        shutil.copy2(source["path"], local_source)
        if _sha256_file(local_source) != source["sha256"]:
            raise AwsTrajectoryError("local smoke source checksum mismatch")
        # The extracted files have their own verified manifest, so retaining a second
        # compressed copy of the smoke result would be pure duplication.
        archive.unlink()
        state["cells"][0]["result_manifest"] = manifest
        state["local_log_dir"] = str(smoke_dir)
        _save_campaign(state, data_root, preflight["clients"])
        state["s3_cleanup"] = _attempt_s3_cleanup(
            state, preflight["clients"], _utc_now()
        )
        _save_campaign(state, data_root)
        _write_json_atomic(smoke_dir / "campaign.json", state)
        print(f"AWS smoke passed and the VM terminated: {smoke_dir}")
        return state


def worker_main(job_path: Path) -> int:
    from aws_worker_runtime import WorkerRuntimeError, worker_main as run_worker

    try:
        return run_worker(job_path)
    except WorkerRuntimeError as error:
        raise AwsTrajectoryError(str(error)) from None


def smoke_worker_main(job: dict, job_path: Path) -> int:
    from aws_worker_runtime import smoke_worker_main as run_smoke_worker

    return run_smoke_worker(job, job_path)


def watchdog_failure_main(job_path: Path, reason: str) -> int:
    from aws_worker_runtime import watchdog_failure_main as run_watchdog

    return run_watchdog(job_path, reason)


def _entrypoint() -> int:
    worker = next((arg.split("=", 1)[1] for arg in sys.argv[1:]
                   if arg.startswith("--worker=")), None)
    watchdog = next((arg.split("=", 1)[1] for arg in sys.argv[1:]
                     if arg.startswith("--watchdog-failure=")), None)
    failure_reason = next((arg.split("=", 1)[1] for arg in sys.argv[1:]
                           if arg.startswith("--failure-reason=")), None)
    if worker:
        return worker_main(Path(worker))
    if watchdog:
        if failure_reason not in {"source_checksum_failed", "worker_watchdog_expired"}:
            raise SystemExit("watchdog invocation requires a recognized --failure-reason")
        return watchdog_failure_main(Path(watchdog), failure_reason)
    raise SystemExit("this module is invoked through exp_real_audit_pipeline.py")


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
