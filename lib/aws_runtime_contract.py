"""Shared constants for the AWS controller and on-instance worker runtime."""

AWS_SCHEMA_VERSION = "mats-real-aws-v1"
WORKER_VERSION = "mats-real-worker-v3-p-hacking-prefix"
FAILURE_PACKAGE_SECONDS = 4 * 3600 + 15 * 60
SSM_PARAMETER_NAME = "/mats/environments/api-keys"

WORKER_PIPELINE_SCRIPTS = frozenset({
    "exp_real_audit_pipeline.py",
    "exp_continuation_pipeline.py",
    "exp_multi_agent_pipeline.py",
    "prefixes/exp_ml_prefix.py",
    "prefixes/exp_p_hacking_prefix.py",
})
DEFAULT_WORKER_PIPELINE_SCRIPT = "exp_real_audit_pipeline.py"
WORKER_PREFIX_PAYLOAD_PATH = "/var/lib/mats-worker/prefix.json"
WORKER_ACTIVITY_LOG_PAYLOAD_PATH = "/var/lib/mats-worker/activity-log.json"

DEFAULT_SECRET_NAMES = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "OPENCODE_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_SUBSCRIPTION_CREDENTIALS_JSON_B64",
    "CODEX_ACCESS_TOKEN",
    "CODEX_SUBSCRIPTION_AUTH_JSON_GZIP_B64",
    "CODEX_SUBSCRIPTION_AUTH_JSON_B64",
)
