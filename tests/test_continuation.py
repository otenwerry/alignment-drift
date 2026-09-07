"""Free tests for the continuation experiment: splicing, judge scoping, payload
validation, task construction, AWS cells, and the viewer's continuations routing."""

from __future__ import annotations

import json
import base64
import hashlib
import io
import os
import pathlib
import sqlite3
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from continuation_evidence import (  # noqa: E402
    CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
    CONTINUATION_PIVOT_PATH,
    CONTINUATION_TASK_DESCRIPTION_PLACEHOLDER,
    ContinuationRun,
    continuation_caveat_fields,
    continuation_record,
    inject_pivot,
    judge_view_messages,
    pivot_preamble,
    prefix_boundary_index,
    reject_dangling_tool_calls,
    seed_family_task_description,
    sliced_message_record,
    update_production_boundary,
)
from environment_judge import EvidenceCaveat  # noqa: E402
from protocol_sources import global_system_prompt  # noqa: E402
import environment_judge.exp_real as exp_real  # noqa: E402
import exp_real_continuation as continuation  # noqa: E402
from exp_target_harness import (  # noqa: E402
    INSPECT_SWE_VERSION,
    NATIVE_RESUME_FORMAT,
    _OPENCODE_REBASE_MESSAGE_IDS_CODE,
    PRODUCTION_SCAFFOLD_VERSIONS,
    production_agent_input_messages,
    retain_native_scaffold_system_messages,
)
from multi_agent_evidence import extract_inline_activity_log  # noqa: E402


SYSTEM_PROMPT = global_system_prompt()


def test_continuation_pipeline_requires_explicit_harness(monkeypatch) -> None:
    sys.path.insert(0, str(ENVIRONMENTS))
    import exp_continuation_pipeline as pipeline

    monkeypatch.delenv("ENVIRONMENTS_JUDGE", raising=False)
    argv = [
        "exp_continuation_pipeline.py",
        "--treatment=test",
        "--prefixes=1",
        "--seed-dir=p_hacking",
        "--seeds=checkout_redesign",
        "--epochs=1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="--harness is required"):
        pipeline._parse_args()
    monkeypatch.setattr(
        sys, "argv", [*argv, "--harness=production", "--pressure=high"]
    )
    default_judge_config = pipeline._parse_args()
    assert default_judge_config["judge_resolved"] == "openai/gpt-5.6-luna"
    assert default_judge_config["gate_model"] == "openai/gpt-5.6-luna"
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--harness=production", "--judge=gpt-5.6-luna"],
    )
    default_config = pipeline._parse_args()
    assert default_config["pressure"] == "low"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--harness=production",
            "--judge=gpt-5.6-luna",
            "--pressure=high",
            "--allow-incomplete-prefixes",
        ],
    )
    config = pipeline._parse_args()
    assert config["harness"] == "production"
    assert config["pressure"] == "high"
    assert config["compute"] == "aws"
    assert config["allow_incomplete_prefixes"] is True


def test_prefix_only_family_cannot_be_a_continuation_destination(monkeypatch) -> None:
    import exp_continuation_pipeline as pipeline

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exp_continuation_pipeline.py",
            "--treatment=no-honeypot",
            "--prefixes=1",
            "--seed-dir=ml_prefix_only",
            "--seeds=demand_forecasting",
            "--epochs=1",
            "--harness=simple",
        ],
    )
    with pytest.raises(SystemExit, match="prefix source"):
        pipeline._parse_args()


def prefix_messages(*, dangling_tool_call: bool = False, trailing_user: bool = False):
    messages = [
        ChatMessageSystem(content=SYSTEM_PROMPT),
        ChatMessageUser(content="Prior task: answer some questions."),
        ChatMessageAssistant(
            content="Working on it.",
            tool_calls=(
                [ToolCall(id="t1", function="bash", arguments={"command": "ls"})]
                if dangling_tool_call else None
            ),
        ),
    ]
    if trailing_user:
        messages.append(ChatMessageUser(content="One more thing."))
    return messages


def payload(messages=None, **overrides) -> dict:
    base = {
        "format": continuation.PREFIX_FORMAT,
        "name": "test-prefix",
        "target": "qwen3-32b",
        "reasoning": False,
        "source": {"kind": "external", "description": "test payload"},
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in (messages if messages is not None else prefix_messages())
        ],
    }
    base.update(overrides)
    return base


def record_for(messages, boundary):
    return continuation_record(
        treatment="full-hack",
        prefix={"name": "test-prefix"},
        prefix_length=len(messages),
        boundary_index=boundary,
        pivot_preamble_text=pivot_preamble(),
        opening_user_message="New task opening.",
        cross_family=True,
    )


def production_native_resume() -> dict:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        content = b"native opencode session"
        member = tarfile.TarInfo("root/.local/share/opencode/opencode.db")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    archive_data = data.getvalue()
    return {
        "format": NATIVE_RESUME_FORMAT,
        "scaffold": "opencode",
        "scaffold_version": PRODUCTION_SCAFFOLD_VERSIONS["opencode"],
        "inspect_swe_version": INSPECT_SWE_VERSION,
        "target_name": "qwen3-32b",
        "routed_slug": "openrouter/qwen/qwen3-32b",
        "reasoning": False,
        "working_directory": "/workspace",
        "workspace_restored": False,
        "native_session_id": None,
        "archive_format": "tar.gz",
        "archive_sha256": hashlib.sha256(archive_data).hexdigest(),
        "archive_bytes": len(archive_data),
        "archive_members": ["root/.local/share/opencode/opencode.db"],
        "archive_base64": base64.b64encode(archive_data).decode("ascii"),
    }


# --------------------------------------------------------------------------- #
# Splicing mechanics
# --------------------------------------------------------------------------- #
def test_continuation_pivot_wording_comes_from_pinned_text_sources() -> None:
    template = CONTINUATION_PIVOT_PATH.read_text().strip()
    assert template.count(CONTINUATION_TASK_DESCRIPTION_PLACEHOLDER) == 1
    assert pivot_preamble() == template.replace(
        CONTINUATION_TASK_DESCRIPTION_PLACEHOLDER, ""
    )
    ml_description = seed_family_task_description("ml_training_data_misuse")
    assert pivot_preamble(ml_description) == template.replace(
        CONTINUATION_TASK_DESCRIPTION_PLACEHOLDER,
        f": {ml_description} this time",
    )


def test_dangling_tool_calls_are_rejected_without_inventing_results() -> None:
    with pytest.raises(ValueError) as exc_info:
        reject_dangling_tool_calls(prefix_messages(dangling_tool_call=True))
    message = str(exc_info.value)
    assert "unresolved tool call" in message
    assert "bash" in message
    assert "id=t1" in message
    reject_dangling_tool_calls(prefix_messages())


def test_pivot_appends_after_an_assistant_ending() -> None:
    messages, boundary = inject_pivot(
        prefix_messages(), pivot_preamble(), "New task opening."
    )
    assert boundary == 3 and len(messages) == 4
    assert messages[boundary].role == "user"
    assert pivot_preamble() in messages[boundary].text
    assert "New task opening." in messages[boundary].text


def test_pivot_folds_into_a_trailing_user_message() -> None:
    original = prefix_messages(trailing_user=True)
    messages, boundary = inject_pivot(
        list(original), pivot_preamble(), "New task opening."
    )
    assert boundary == len(original) - 1 and len(messages) == len(original)
    text = messages[boundary].text
    assert "One more thing." in text and "New task opening." in text


def test_folded_prior_user_text_is_hidden_from_judge_but_retained_for_target() -> None:
    prefix = prefix_messages(trailing_user=True)
    messages, boundary = inject_pivot(
        list(prefix), pivot_preamble(), "New task opening."
    )
    messages.append(ChatMessageAssistant(content="Live answer."))
    record = record_for(prefix, boundary)

    assert "One more thing." in messages[boundary].text
    judged = judge_view_messages(messages, record)
    assert judged[1].id == messages[boundary].id
    assert "One more thing." not in judged[1].text
    assert judged[1].text == f"{pivot_preamble()}\n\nNew task opening."
    assert sliced_message_record(record, len(messages))[
        "boundary_message_prefix_text_excluded"
    ] is True
    assert "shared the hand-off user message" in continuation_caveat_fields(record)[
        "description"
    ]


def test_judge_view_keeps_system_plus_live_task_only() -> None:
    prefix = prefix_messages()
    messages, boundary = inject_pivot(list(prefix), pivot_preamble(), "New task.")
    messages.append(ChatMessageAssistant(content="Live answer."))
    record = record_for(prefix, boundary)
    judged = judge_view_messages(messages, record)
    texts = [message.text or "" for message in judged]
    assert judged[0].role == "system"
    assert len(judged) == 3
    assert not any("Prior task" in text for text in texts)
    assert judged[1].id == messages[boundary].id  # citation ids survive the slice
    scope = sliced_message_record(record, len(messages))
    assert scope == {
        "prefix_messages_excluded": boundary - 1,
        "judged_message_count": len(messages) - boundary + 1,
        "total_message_count": len(messages),
    }


def test_judge_view_rejects_a_broken_boundary() -> None:
    prefix = prefix_messages()
    messages, boundary = inject_pivot(list(prefix), pivot_preamble(), "New task.")
    with pytest.raises(ValueError):
        judge_view_messages(messages, record_for(prefix, len(messages)))
    with pytest.raises(ValueError):
        judge_view_messages(messages[1:], record_for(prefix, boundary - 1))


def test_continuation_caveat_is_a_valid_evidence_caveat() -> None:
    record = record_for(prefix_messages(), 3)
    caveat = EvidenceCaveat(**continuation_caveat_fields(record))
    assert caveat.code == "prior_unrelated_task_omitted"
    assert "2 source message(s)" in caveat.description


def test_production_native_resume_keeps_scaffold_systems_for_judge() -> None:
    prefix = [
        ChatMessageSystem(content="Production scaffold system A"),
        ChatMessageSystem(content="Production scaffold system B"),
        ChatMessageUser(content="Prior task."),
        ChatMessageAssistant(content="Prior answer."),
    ]
    saved, boundary = inject_pivot(
        prefix, pivot_preamble(), "New task opening."
    )
    saved.append(
        ChatMessageAssistant(content="Live answer."),
    )
    record = record_for(prefix, boundary)
    record["production_native_resume"] = {
        "archive_sha256": "abc",
        "scaffold_system_message_count": 2,
    }

    judged = judge_view_messages(saved, record)
    assert [message.role for message in judged] == [
        "system", "system", "user", "assistant",
    ]
    assert not any("Prior task" in (message.text or "") for message in judged)
    assert judged[2].text == f"{pivot_preamble()}\n\nNew task opening."
    scope = sliced_message_record(record, len(saved))
    assert scope["production_prefix_resumed_natively"] is True
    assert scope["production_old_workspace_restored"] is False
    assert scope["production_scaffold_system_messages_retained"] == 2
    assert "resumed its native session state" in continuation_caveat_fields(record)["description"]


def test_native_inline_context_starts_fresh_and_is_hidden_from_judge() -> None:
    prefix = [ChatMessageUser(content="Complete prior activity log.")]
    record = continuation_record(
        treatment="activity-log-context",
        prefix={"name": "activity-log-traj711", "source_harness": "production"},
        prefix_length=1,
        boundary_index=0,
        pivot_preamble_text=pivot_preamble(),
        opening_user_message="New task opening.",
        cross_family=True,
        delivery_mode=CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
    )
    run = ContinuationRun(
        prefix_messages=tuple(prefix),
        pivot_text=pivot_preamble(),
        record=record,
        delivery_mode=CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
    )
    initial, boundary = run.initial_messages()
    assert boundary == 0
    assert "Complete prior activity log." in initial[0].text

    returned = [
        ChatMessageSystem(content="Native scaffold system"),
        initial[0],
        ChatMessageAssistant(content="Live answer."),
    ]
    record["production_fresh_context"] = {
        "delivery_mode": CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
        "fresh_session": True,
    }
    assert update_production_boundary(returned, record) == 1
    judged = judge_view_messages(returned, record)
    assert [message.role for message in judged] == ["system", "user", "assistant"]
    assert "Complete prior activity log." not in judged[1].text
    assert judged[1].text == f"{pivot_preamble()}\n\nNew task opening."
    scope = sliced_message_record(record, len(returned))
    assert scope["native_prefix_resumed"] is False
    assert scope["native_inline_user_context"] is True
    assert scope["boundary_message_prefix_text_excluded"] is True
    caveat = continuation_caveat_fields(record)
    assert "activity log" in caveat["description"]


def test_production_boundary_accepts_opencode_outer_quote_wrapper() -> None:
    prefix = [
        ChatMessageSystem(content="Production scaffold system"),
        ChatMessageUser(content='"Prior task."'),
        ChatMessageAssistant(content="Prior answer."),
    ]
    saved, boundary = inject_pivot(
        prefix, pivot_preamble(), "New task opening."
    )
    saved[boundary] = saved[boundary].model_copy(
        update={"content": f'"{saved[boundary].text}"'}, deep=True
    )
    saved.append(ChatMessageAssistant(content="Live answer."))
    record = record_for(prefix, boundary)
    record["production_native_resume"] = {"archive_sha256": "abc"}

    assert update_production_boundary(saved, record) == boundary
    native = record["production_native_resume"]
    assert native["boundary_lost"] is False
    assert native["boundary_match_count"] == 1
    assert native["boundary_user_text_wrapper"] == "outer_double_quotes"

    judged = judge_view_messages(saved, record)
    assert [message.role for message in judged] == [
        "system", "user", "assistant",
    ]
    assert judged[1].text == (
        f'"{pivot_preamble()}\n\nNew task opening."'
    )


def test_production_boundary_still_fails_closed_without_exact_handoff() -> None:
    messages = [
        ChatMessageSystem(content="Production scaffold system"),
        ChatMessageUser(content='"A different task."'),
    ]
    record = record_for(prefix_messages(), 3)
    record["production_native_resume"] = {"archive_sha256": "abc"}

    with pytest.raises(RuntimeError, match=r"matches=\[\]"):
        update_production_boundary(messages, record)
    assert record["production_native_resume"]["boundary_lost"] is True


def test_production_boundary_fails_closed_without_handoff_response() -> None:
    messages = [
        ChatMessageSystem(content="Production scaffold system"),
        ChatMessageUser(content=f'"{pivot_preamble()}\n\nNew task opening."'),
    ]
    record = record_for(prefix_messages(), 3)
    record["production_native_resume"] = {"archive_sha256": "abc"}

    with pytest.raises(RuntimeError, match="without an assistant response"):
        update_production_boundary(messages, record)
    native = record["production_native_resume"]
    assert native["handoff_response_missing"] is True
    assert native["handoff_assistant_message_count"] == 0


def test_native_resume_does_not_reinject_scaffold_system_as_a_new_instruction() -> None:
    messages = [
        ChatMessageSystem(content="Native scaffold system"),
        ChatMessageUser(content="Old task"),
        ChatMessageAssistant(content="Old answer"),
        ChatMessageUser(content="New task"),
    ]
    resumed = production_agent_input_messages(messages)
    assert [message.role for message in resumed] == ["user", "assistant", "user"]
    fresh = production_agent_input_messages([ChatMessageUser(content="First task")])
    assert [message.role for message in fresh] == ["user"]


def test_native_resume_retains_scaffold_system_only_in_inspect_history() -> None:
    prefix = [
        ChatMessageSystem(content="Native scaffold system"),
        ChatMessageUser(content='"Old task"'),
        ChatMessageAssistant(content="Old answer"),
    ]
    saved, boundary = inject_pivot(
        prefix, pivot_preamble(), "New task opening."
    )
    native_input = production_agent_input_messages(saved)
    assert not any(message.role == "system" for message in native_input)

    returned = [*native_input, ChatMessageAssistant(content="Live answer")]
    retained = retain_native_scaffold_system_messages(saved, returned)
    record = record_for(prefix, boundary)
    record["production_native_resume"] = {"archive_sha256": "abc"}

    assert retained[0] is saved[0]
    assert update_production_boundary(retained, record) == boundary
    assert [message.role for message in judge_view_messages(retained, record)] == [
        "system", "user", "assistant",
    ]


def test_native_resume_does_not_duplicate_returned_scaffold_system() -> None:
    previous = [
        ChatMessageSystem(content="Native scaffold system"),
        ChatMessageUser(content="Old task"),
    ]
    returned = [
        ChatMessageSystem(content="Returned scaffold system"),
        ChatMessageUser(content="New task"),
    ]
    assert retain_native_scaffold_system_messages(previous, returned) == returned


def test_opencode_resume_message_ids_are_rebased_in_transcript_order(
    tmp_path,
) -> None:
    database = tmp_path / "opencode.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, data TEXT
        );
        CREATE TABLE event (data TEXT);
        INSERT INTO message VALUES
            ('msg_fff000000001aaaaaaaaaaaaaa', 's1', 10,
             '{"role":"user"}'),
            ('msg_fff000000002bbbbbbbbbbbbbb', 's1', 20,
             '{"role":"assistant","parentID":"msg_fff000000001aaaaaaaaaaaaaa"}');
        INSERT INTO part VALUES
            ('prt_1', 'msg_fff000000001aaaaaaaaaaaaaa', '{"type":"text"}'),
            ('prt_2', 'msg_fff000000002bbbbbbbbbbbbbb', '{"type":"text"}');
        INSERT INTO event VALUES
            ('{"messageID":"msg_fff000000002bbbbbbbbbbbbbb"}');
        """
    )
    connection.commit()
    connection.close()
    environment = dict(os.environ)
    environment["MATS_OPENCODE_DB_PATH"] = str(database)

    result = subprocess.run(
        [sys.executable, "-c", _OPENCODE_REBASE_MESSAGE_IDS_CODE],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(result.stdout)
    assert metadata["applied"] is True
    assert metadata["message_count"] == 2

    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT id, data FROM message ORDER BY time_created, id"
    ).fetchall()
    ids = [row[0] for row in rows]
    assert ids == sorted(ids)
    assert all(len(message_id) == 30 for message_id in ids)
    assert json.loads(rows[1][1])["parentID"] == ids[0]
    assert connection.execute(
        "SELECT message_id FROM part ORDER BY id"
    ).fetchall() == [(ids[0],), (ids[1],)]
    assert json.loads(connection.execute("SELECT data FROM event").fetchone()[0])[
        "messageID"
    ] == ids[1]
    connection.close()


def test_continuation_run_epochs_get_independent_copies() -> None:
    prefix = tuple(prefix_messages())
    record = record_for(list(prefix), 3)
    run = ContinuationRun(
        prefix_messages=prefix, pivot_text=pivot_preamble(), record=record
    )
    first, boundary_a = run.initial_messages()
    second, boundary_b = run.initial_messages()
    assert boundary_a == boundary_b == record["boundary_index"]
    assert first[1] is not second[1]
    first[1].content = "mutated"
    assert second[1].text == "Prior task: answer some questions."


# --------------------------------------------------------------------------- #
# Shared final-judge path slices and declares the omission
# --------------------------------------------------------------------------- #
def test_final_judge_call_slices_the_prefix_and_adds_the_caveat(monkeypatch) -> None:
    prefix = prefix_messages()
    messages, boundary = inject_pivot(list(prefix), pivot_preamble(), "New task.")
    messages.append(ChatMessageAssistant(content="Live answer."))
    record = record_for(prefix, boundary)
    seen: dict = {}

    async def fake_judge_trajectory(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(value={}, metadata={})

    monkeypatch.setattr(exp_real, "judge_trajectory", fake_judge_trajectory)
    import asyncio
    asyncio.run(exp_real.judge_complete_real_trajectory(
        family="p_hacking",
        messages=messages,
        real_env={"continuation": record},
        model=None,
    ))
    judged_texts = [message.text or "" for message in seen["messages"]]
    assert len(seen["messages"]) == 3
    assert not any("Prior task" in text for text in judged_texts)
    codes = [caveat.code for caveat in seen["upstream_caveats"]]
    assert "prior_unrelated_task_omitted" in codes


def test_ordinary_final_judge_call_is_untouched(monkeypatch) -> None:
    messages = [
        ChatMessageSystem(content=SYSTEM_PROMPT),
        ChatMessageUser(content="Task."),
        ChatMessageAssistant(content="Answer."),
    ]
    seen: dict = {}

    async def fake_judge_trajectory(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(value={}, metadata={})

    monkeypatch.setattr(exp_real, "judge_trajectory", fake_judge_trajectory)
    import asyncio
    asyncio.run(exp_real.judge_complete_real_trajectory(
        family="p_hacking", messages=messages, real_env={}, model=None,
    ))
    assert len(seen["messages"]) == 3
    codes = [caveat.code for caveat in seen["upstream_caveats"]]
    assert "prior_unrelated_task_omitted" not in codes


# --------------------------------------------------------------------------- #
# Prefix payload validation
# --------------------------------------------------------------------------- #
def test_payload_validation_rejects_bad_shapes() -> None:
    good = payload()
    for broken in (
        {**good, "format": "other"},
        {**good, "name": "Bad Name"},
        {**good, "target": "not-a-model"},
        {**good, "reasoning": "yes"},
        {**good, "source": {"kind": "mystery"}},
        {**good, "messages": []},
    ):
        with pytest.raises(SystemExit):
            continuation.validate_prefix_payload(broken, origin="test")
    assert continuation.validate_prefix_payload(good, origin="test") is good


def test_prefix_spec_requires_the_current_system_prompt() -> None:
    wrong = prefix_messages()
    wrong[0] = ChatMessageSystem(content="a completely different prompt")
    with pytest.raises(SystemExit) as excinfo:
        continuation.build_prefix_spec(payload(messages=wrong))
    assert "SYSTEM_PROMPT" in str(excinfo.value)


def test_prefix_spec_inserts_the_prompt_when_absent_and_flags_it() -> None:
    spec = continuation.build_prefix_spec(payload(messages=prefix_messages()[1:]))
    assert spec.system_prompt_inserted
    assert spec.messages[0].role == "system"
    assert spec.messages[0].text == SYSTEM_PROMPT
    assert spec.boundary_index == len(spec.messages)


def test_activity_log_prefix_is_free_static_user_context_for_any_harness() -> None:
    source = payload(
        messages=prefix_messages(),
        source={
            "kind": "trajectory",
            "trajectory_id": 711,
            "run": "production-run",
            "task": "real_audit_qwen3-32b_reasoning_prompt_benchmark",
            "seed": "reasoning_prompt_benchmark",
            "epoch": 1,
            "family": "p_hacking",
            "harness": "production",
        },
    )
    built = continuation.activity_log_prefix_payload_from_trajectory(source)
    rebuilt = continuation.activity_log_prefix_payload_from_trajectory(source)

    assert built == rebuilt
    assert continuation.payload_sha256(built) == continuation.payload_sha256(rebuilt)
    assert built["delivery"]["mode"] == CONTINUATION_DELIVERY_INLINE_USER_CONTEXT
    assert "native_resume" not in built
    assert [message["role"] for message in built["messages"]] == ["user"]
    content = extract_inline_activity_log(built["messages"][0]["content"])
    metadata = built["source"]["activity_log_metadata"]
    assert len(content.splitlines()) == metadata["line_count"]
    assert "Prior task: answer some questions." in content
    assert SYSTEM_PROMPT not in content

    production = continuation.build_prefix_spec(built, harness="production")
    subscription = continuation.build_prefix_spec(built, harness="subscription")
    simple = continuation.build_prefix_spec(built, harness="simple")
    assert production.native_resume is subscription.native_resume is None
    assert [message.role for message in production.messages] == ["user"]
    assert production.boundary_index == 0
    assert [message.role for message in simple.messages] == ["system", "user"]
    assert simple.boundary_index == 1


def test_external_ml_prefix_becomes_self_contained_activity_log_context() -> None:
    source = payload(
        name="demand-prefix-e1",
        messages=prefix_messages(),
        source={
            "kind": "external",
            "family": "ml_prefix_only",
            "comparison_source_family": "ml_training_data_misuse",
            "seed": "demand_forecasting",
            "epoch": 1,
            "harness": "production",
            "generator": "exp_ml_prefix.py",
            "prefix_type": "ml_prefix_only",
            "prefix_type_label": "ML prefix only",
            "source_label": "Demand forecasting · no test access",
            "completed_protocol": True,
            "continuation_eligibility": {
                "eligible_by_default": True,
                "missing_required_deliverables": [],
            },
        },
        native_resume=production_native_resume(),
    )

    built = continuation.activity_log_prefix_payload_from_source(source)
    source_identity = built["source"]["activity_log_source_payload"]

    assert built["name"] == "activity-log-demand-prefix-e1"
    assert built["source"]["kind"] == "external"
    assert built["source"]["family"] == "ml_prefix_only"
    assert built["source"]["comparison_source_family"] == (
        "ml_training_data_misuse"
    )
    assert built["source"]["prefix_type"] == "activity_log_context"
    assert built["source"]["source_label"] == (
        "Demand forecasting · no test access"
    )
    assert source_identity == {
        "kind": "continuation_prefix_payload",
        "format": continuation.PREFIX_FORMAT,
        "name": "demand-prefix-e1",
        "sha256": continuation.payload_sha256(source),
        "prefix_type": "ml_prefix_only",
        "prefix_type_label": "ML prefix only",
        "generator": "exp_ml_prefix.py",
    }
    assert "native_resume" not in built
    assert continuation.build_prefix_spec(
        built, harness="production"
    ).native_resume is None


def test_activity_log_prefix_rejects_nested_activity_log_payload() -> None:
    source = payload(
        source={
            "kind": "external",
            "prefix_type": "activity_log_context",
        }
    )
    with pytest.raises(SystemExit, match="second time"):
        continuation.activity_log_prefix_payload_from_source(source)


def test_activity_log_prefix_rejects_tampered_inline_content() -> None:
    source = payload(
        source={"kind": "trajectory", "trajectory_id": 711},
    )
    built = continuation.activity_log_prefix_payload_from_trajectory(source)
    built["messages"][0]["content"] += "tampered"
    with pytest.raises(SystemExit, match="pinned wrapper"):
        continuation.build_prefix_spec(built, harness="production")


def test_prefix_spec_rejects_mid_conversation_system_messages() -> None:
    broken = prefix_messages() + [ChatMessageSystem(content="another")]
    with pytest.raises(SystemExit):
        continuation.build_prefix_spec(payload(messages=broken))


def test_production_prefix_can_reuse_a_production_scaffold_transcript() -> None:
    production_messages = [
        ChatMessageSystem(content=f"Claude Code system\n\n{SYSTEM_PROMPT}"),
        ChatMessageUser(content="Prior task."),
        ChatMessageAssistant(content="Prior answer."),
        ChatMessageSystem(content="Compaction marker"),
    ]
    raw = payload(
        messages=production_messages,
        source={
            "kind": "trajectory",
            "harness": "production",
            "seed": "retrieval_practice",
        },
        native_resume=production_native_resume(),
    )
    with pytest.raises(SystemExit, match="--harness=production"):
        continuation.build_prefix_spec(raw, harness="simple")
    with pytest.raises(SystemExit, match="matching subscription-scaffold"):
        continuation.build_prefix_spec(raw, harness="subscription")
    spec = continuation.build_prefix_spec(raw, harness="production")
    assert len(spec.messages) == len(production_messages)
    assert spec.messages[3].role == "system"


def test_production_prefix_rejects_a_transcript_without_native_state() -> None:
    raw = payload(
        source={"kind": "external", "harness": "production"},
    )
    with pytest.raises(SystemExit, match="native_resume bundle"):
        continuation.build_prefix_spec(raw, harness="production")


def test_subscription_prefix_requires_matching_subscription_native_state() -> None:
    native_messages = [
        ChatMessageSystem(content="Native scaffold system"),
        ChatMessageUser(content="Prior task."),
        ChatMessageAssistant(content="Prior answer."),
    ]
    raw = payload(
        messages=native_messages,
        source={"kind": "external", "harness": "subscription"},
        native_resume=production_native_resume(),
    )
    spec = continuation.build_prefix_spec(raw, harness="subscription")
    assert spec.native_resume is raw["native_resume"]
    with pytest.raises(SystemExit, match="matching production-scaffold"):
        continuation.build_prefix_spec(raw, harness="production")


def test_prefix_spec_fails_loudly_on_dangling_tool_calls() -> None:
    with pytest.raises(SystemExit) as exc_info:
        continuation.build_prefix_spec(
            payload(messages=prefix_messages(dangling_tool_call=True))
        )
    message = str(exc_info.value)
    assert "prefix 'test-prefix'" in message
    assert "unresolved tool call" in message
    assert "bash" in message
    assert "id=t1" in message


def test_prefix_spec_assigns_missing_message_ids() -> None:
    raw = payload()
    for message in raw["messages"]:
        message.pop("id", None)
    spec = continuation.build_prefix_spec(raw)
    ids = [message.id for message in spec.messages]
    assert all(ids) and len(ids) == len(set(ids))


def test_one_turn_cutoff_is_explicit_and_queryable() -> None:
    raw = payload(
        messages=[
            ChatMessageSystem(content=SYSTEM_PROMPT),
            ChatMessageUser(content="First pass."),
            ChatMessageAssistant(content="First result."),
            ChatMessageUser(content="Second pass."),
            ChatMessageAssistant(content="Second result."),
        ],
        source={
            "kind": "trajectory",
            "trajectory_id": 9,
            "harness": "simple",
            "prefix_condition": {
                "key": "hack_2turn",
                "label": "hack · 2 turns",
            },
            "first_pass_condition": {
                "key": "notable",
                "label": "notable · answer-key peek",
            },
        },
    )

    derived = continuation.cutoff_prefix_payload(
        raw, before_experiment_user_turn=2
    )

    assert derived["name"] == "test-prefix-cutoff-u2"
    assert [message["role"] for message in derived["messages"]] == [
        "system", "user", "assistant",
    ]
    assert derived["source"]["prefix_condition"] == {
        "key": "notable_1turn_cutoff",
        "label": "notable · answer-key peek · 1-turn cutoff",
    }
    cutoff = derived["source"]["cutoff"]
    assert cutoff["affected"] is True
    assert cutoff["before_experiment_user_turn"] == 2
    assert cutoff["retained_message_count"] == 3
    assert cutoff["omitted_message_count"] == 2
    assert cutoff["native_state_rewritten"] is False
    assert derived["source"]["lossy_processing"]["affected"] is True


def test_opencode_cutoff_rewinds_sqlite_state_and_repacks_allowed_roots(
    tmp_path,
) -> None:
    root = tmp_path / "root"
    database = root / ".local" / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    locks = root / ".local" / "state" / "opencode" / "locks"
    locks.mkdir(parents=True)
    (locks / "stale.lock").write_text("locked")
    log = database.parent / "log" / "opencode.log"
    log.parent.mkdir()
    log.write_text("Second pass.")

    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, time_created INTEGER, time_updated INTEGER
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,
            time_updated INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, data TEXT
        );
        CREATE TABLE event (
            aggregate_id TEXT, seq INTEGER, data TEXT
        );
        CREATE TABLE event_sequence (aggregate_id TEXT PRIMARY KEY, seq INTEGER);
        CREATE TABLE todo (
            id TEXT PRIMARY KEY, time_created INTEGER, time_updated INTEGER
        );
        INSERT INTO session VALUES ('s1', 1, 40);
        INSERT INTO message VALUES
            ('u1', 's1', 10, 10, '{"role":"user"}'),
            ('a1', 's1', 20, 20, '{"role":"assistant"}'),
            ('u2', 's1', 30, 30, '{"role":"user"}'),
            ('a2', 's1', 40, 40, '{"role":"assistant"}');
        INSERT INTO part VALUES
            ('p1', 'u1', '{"type":"text","text":"First pass."}'),
            ('p2', 'a1', '{"type":"text","text":"First result."}'),
            ('p3', 'u2', '{"type":"text","text":"Second pass."}'),
            ('p4', 'a2', '{"type":"text","text":"Second result."}');
        INSERT INTO event VALUES
            ('s1', 1, '{"message_id":"u1"}'),
            ('s1', 2, '{"message_id":"a1"}'),
            ('s1', 3, '{"message_id":"u2"}'),
            ('s1', 4, '{"message_id":"a2"}');
        INSERT INTO event_sequence VALUES ('s1', 4);
        INSERT INTO todo VALUES ('before', 10, 20), ('after', 10, 35);
        """
    )
    connection.commit()
    connection.close()

    result = continuation._rewrite_opencode_cutoff(
        root, cutoff_user_text="Second pass."
    )

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT count(*) FROM message").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM part").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM event").fetchone()[0] == 2
    assert connection.execute("SELECT seq FROM event_sequence").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM todo").fetchone()[0] == 1
    assert connection.execute("SELECT time_updated FROM session").fetchone()[0] == 20
    connection.close()
    assert result["omitted_native_messages"] == 2
    assert result["omitted_native_events"] == 2
    assert not log.exists()
    assert list(locks.iterdir()) == []

    archive_data = continuation._repack_native_archive(
        tmp_path, scaffold="opencode"
    )
    members = continuation._archive_member_names(archive_data)
    assert members
    assert all(
        member == "root/.local/share/opencode"
        or member.startswith("root/.local/share/opencode/")
        or member == "root/.local/state/opencode"
        or member.startswith("root/.local/state/opencode/")
        for member in members
    )
    assert "root" not in members
    assert "root/.local" not in members


def test_codex_cutoff_retains_the_completed_first_invocation(tmp_path) -> None:
    root = tmp_path / "root"
    rollout = root / ".codex" / "sessions" / "rollout-test.jsonl"
    rollout.parent.mkdir(parents=True)
    records = [
        {"type": "turn_context", "payload": {}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "First pass."}],
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [],
        }},
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        {"type": "thread_settings", "payload": {}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Second pass."}],
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [],
        }},
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    rollout.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    result = continuation._rewrite_codex_cutoff(
        root, cutoff_user_text="Second pass."
    )

    retained = [json.loads(line) for line in rollout.read_text().splitlines()]
    assert retained == records[:4]
    assert result["retained_native_events"] == 4
    assert result["omitted_native_events"] == 4


def test_prefix_file_only_load_does_not_require_trajectory_registry(
    tmp_path, monkeypatch
) -> None:
    payload_path = tmp_path / "prefix.json"
    payload_path.write_text(json.dumps(payload()))
    monkeypatch.setattr(
        continuation, "REGISTRY_FILE", tmp_path / "missing-trajectory-ids.json"
    )

    specs = continuation.load_prefix_specs(
        [], [str(payload_path)], harness="simple"
    )

    assert [spec.name for spec in specs] == ["test-prefix"]


def test_old_trajectory_payload_is_rechecked_when_local_registry_exists(
    tmp_path, monkeypatch
) -> None:
    raw = payload(source={"kind": "trajectory", "trajectory_id": 9})
    payload_path = tmp_path / "prefix.json"
    payload_path.write_text(json.dumps(raw))
    registry = tmp_path / "trajectory-ids.json"
    registry.write_text("{}")
    monkeypatch.setattr(continuation, "REGISTRY_FILE", registry)
    monkeypatch.setattr(
        continuation,
        "_registry_entries",
        lambda ids: {ids[0]: {
            "mode": "run", "task": "task", "seed": "seed", "epoch": 1,
        }},
    )

    def reject(*_args, **_kwargs):
        raise SystemExit(
            "trajectory #9 is benchmark only; refusing it as a prefix"
        )

    monkeypatch.setattr(
        continuation,
        "reconstruct_prefix_payload",
        reject,
    )

    with pytest.raises(SystemExit, match="benchmark only"):
        continuation.load_prefix_specs([], [str(payload_path)], harness="simple")


def test_trajectory_payload_fails_closed_without_local_registry(
    tmp_path, monkeypatch
) -> None:
    raw = payload(source={"kind": "trajectory", "trajectory_id": 9})
    payload_path = tmp_path / "prefix.json"
    payload_path.write_text(json.dumps(raw))
    monkeypatch.setattr(
        continuation, "REGISTRY_FILE", tmp_path / "missing-trajectory-ids.json"
    )
    monkeypatch.delenv("MATS_REMOTE_RUN_DIR", raising=False)

    with pytest.raises(SystemExit, match="cannot verify"):
        continuation.load_prefix_specs([], [str(payload_path)], harness="simple")


def test_treatment_slug_validation() -> None:
    assert continuation.validate_treatment(" Full-Hack ") == "full-hack"
    for bad in ("full_hack", "full hack", "-full", ""):
        with pytest.raises(SystemExit):
            continuation.validate_treatment(bad)


# --------------------------------------------------------------------------- #
# Cells and tasks
# --------------------------------------------------------------------------- #
def spec_for(**overrides):
    return continuation.build_prefix_spec(payload(**overrides))


def test_cells_cross_product_and_family_descriptors() -> None:
    spec = spec_for()
    cells = continuation.build_continuation_cells(
        [spec], str(ENVIRONMENTS / "seeds" / "p_hacking"),
        ["checkout_redesign", "retrieval_practice"],
    )
    assert [cell.member for cell in cells] == [
        "checkout_redesign", "retrieval_practice",
    ]
    assert all(cell.cross_family is None for cell in cells)  # external prefix


def test_cells_refuse_the_prefix_own_seed() -> None:
    trajectory_payload = payload(
        source={"kind": "trajectory", "trajectory_id": 9,
                "seed": "checkout_redesign", "family": "p_hacking"},
    )
    spec = continuation.build_prefix_spec(trajectory_payload)
    with pytest.raises(SystemExit) as excinfo:
        continuation.build_continuation_cells(
            [spec], str(ENVIRONMENTS / "seeds" / "p_hacking"),
            ["checkout_redesign"],
        )
    assert "different seed" in str(excinfo.value)
    cells = continuation.build_continuation_cells(
        [spec], str(ENVIRONMENTS / "seeds" / "p_hacking"), ["retrieval_practice"],
    )
    assert cells[0].cross_family is False  # same family, different seed


def test_trajectory_prefix_preflight_rejects_benchmark_only_timeout(
    monkeypatch, tmp_path,
) -> None:
    import inspect_ai.log

    mode_dir = tmp_path / "timed-run"
    mode_dir.mkdir()
    monkeypatch.setattr(continuation, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(inspect_ai.log, "list_eval_logs", lambda _path: ["run.eval"])

    sample = SimpleNamespace(
        id="fraud_detection",
        epoch=1,
        metadata={"real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
            "gates": [],
        }},
        role_usage={"target": SimpleNamespace(output_tokens=12)},
        messages=[SimpleNamespace(
            role="assistant", content="partial work", tool_calls=[]
        )],
        events=[],
        scores={"environment_judge": SimpleNamespace(metadata={
            "environment_judge": {
                "result": {"environment_failure": {"value": False}},
                "post_validation": "passed",
            }
        })},
    )
    header = SimpleNamespace(eval=SimpleNamespace(task="real_audit_test"))
    full = SimpleNamespace(
        eval=SimpleNamespace(
            task="real_audit_test",
            metadata={"target_name": "qwen3-32b", "reasoning": True},
            model_roles={"target": SimpleNamespace(model="mock/target")},
        ),
        samples=[sample],
    )
    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log",
        lambda _path, header_only=False, **_kwargs: header if header_only else full,
    )

    with pytest.raises(SystemExit) as excinfo:
        continuation.reconstruct_prefix_payload(1, {
            "mode": "timed-run",
            "task": "real_audit_test",
            "seed": "fraud_detection",
            "epoch": 1,
        })

    message = str(excinfo.value)
    assert "benchmark only" in message
    assert "time limit" in message
    assert "refusing it as a prefix" in message


def test_task_metadata_records_the_full_continuation_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        continuation, "build_target",
        lambda slug, **kwargs: SimpleNamespace(
            model="mockllm/model", routed_slug=slug, reasoning_enabled=None,
            strict_tools=None, prompt_cache_key=None,
        ),
    )
    spec = spec_for()
    cells = continuation.build_continuation_cells(
        [spec], str(ENVIRONMENTS / "seeds" / "p_hacking"), ["checkout_redesign"],
    )
    tasks = continuation.build_continuation_tasks(
        cells, treatment="full-hack", run_label="test-run", condition="allow",
        gate_model="mockllm/model", judge="mockllm/model", artifacts_root=None,
        harness="simple", pressure="high",
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task.name == (
        "continuation_full-hack_qwen3-32b_checkout_redesign_ptest-prefix"
    )
    metadata = task.metadata
    assert metadata["experiment"] == "continuation"
    assert metadata["treatment"] == "full-hack"
    assert metadata["prefix_name"] == "test-prefix"
    assert metadata["prefix_sha256"] == spec.sha256
    assert metadata["prefix_boundary_index"] == spec.boundary_index
    assert metadata["cross_family"] is None
    assert "next task: an experiment analysis task this time" in (
        metadata["pivot_preamble"]
    )
    assert metadata["target_tools_mode"] == "real"
    assert metadata["condition"] == "allow"
    assert metadata["pressure"] == "high"
    assert task.dataset[0].metadata["protocol"]["pressure"] == "high"


def test_long_task_name_is_shortened_but_full_identity_stays_in_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        continuation, "build_target",
        lambda slug, **kwargs: SimpleNamespace(
            model="mockllm/model", routed_slug=slug, reasoning_enabled=None,
            strict_tools=None, prompt_cache_key=None,
        ),
    )
    prefix_name = "nq1234-2000-deepseek-v4-pro-20260812001909"
    suffix = "purpose-built-prefix-native-fixed-nq1234-2000-de-b948ed03"
    spec = spec_for(name=prefix_name)
    cells = continuation.build_continuation_cells(
        [spec], str(ENVIRONMENTS / "seeds" / "p_hacking"),
        ["checkout_redesign"],
    )
    task = continuation.build_continuation_tasks(
        cells, treatment="purpose-built-prefix-native-fixed",
        run_label="test-run", condition="allow", gate_model="mockllm/model",
        judge="mockllm/model", artifacts_root=None, task_id_suffix=suffix,
        harness="simple", pressure="low",
    )[0]

    assert len(task.name) <= 160
    assert task.name.endswith(suffix)
    assert task.metadata["task_name_shortened"] is True
    assert task.metadata["task_name_full"].startswith(
        "continuation_purpose-built-prefix-native-fixed_"
    )
    assert task.metadata["task_name_full"].endswith(suffix)


# --------------------------------------------------------------------------- #
# AWS continuation cells
# --------------------------------------------------------------------------- #
def test_aws_continuation_cells_carry_script_and_payload_identity() -> None:
    import exp_aws_trajectory as aws

    cfg = {
        "continuation": {
            "treatment": "full-hack",
            "payloads": [{
                "name": "traj9", "sha256": "a" * 64, "file_sha256": "b" * 64,
                "local_path": "/tmp/traj9.json", "target": "qwen3-32b",
                "target_model": "openrouter/qwen/qwen3-32b",
                "reasoning": True,
            }],
        },
        "seeds": ["fraud_detection"],
        "epochs": 2,
        "seeds_path": "/repo/environments/seeds/ml_training_data_misuse",
        "seed_dir": "ml_training_data_misuse",
        "condition": "allow",
        "judge_resolved": "openai/gpt-5.6-luna",
        "gate_model": "openai/gpt-5.6-luna",
        "time_limit": 4200,
        "harness": "simple",
        "allow_incomplete_prefixes": True,
        "aws_region": "us-west-2",
        "aws_instance_type": "c7a.xlarge",
    }
    cells = aws.build_continuation_cells(
        cfg, campaign_id="continuation-aws-test", source={
            "sha256": "c" * 64, "bytes": 1, "files": 1,
        },
        bucket="bucket", hourly_price=0.1,
    )
    assert len(cells) == 2
    cell = cells[0]
    assert cell["pipeline_script"] == "exp_continuation_pipeline.py"
    assert cell["prefix_name"] == "traj9"
    assert cell["prefix_payload_sha256"] == "b" * 64
    assert cell["prefix_sha256"] == "a" * 64
    assert cell["prefix_payload_key"].startswith(
        "continuation-aws-test/prefixes/traj9-aaaaaaaaaaaa"
    ) or cell["prefix_payload_key"] == (
        "campaigns/continuation-aws-test/prefixes/traj9-aaaaaaaaaaaa.json"
    )
    args = cell["pipeline_args"]
    assert "--treatment=full-hack" in args
    assert f"--prefix-files={aws.WORKER_PREFIX_PAYLOAD_PATH}" in args
    assert "--seed-dir=ml_training_data_misuse" in args
    assert "--seeds=fraud_detection" in args
    assert "--compute=local" in args
    assert "--harness=simple" in args
    assert "--skip-viewer" in args
    assert "--allow-incomplete-prefixes" in args
    assert cell["family"] == "ml_training_data_misuse"
    assert cell["sandbox_compose"] == "environments/sandbox/ml/compose.yaml"
    assert cell["task_name"].startswith(
        "continuation_full-hack_qwen3-32b_fraud_detection_ptraj9"
    )

    p_cfg = {
        **cfg,
        "seeds": ["checkout_redesign"],
        "seeds_path": "/repo/environments/seeds/p_hacking",
        "seed_dir": "p_hacking",
        "family": "p_hacking",
        "pressure": "high",
        "time_limit": 1800,
    }
    p_cell = aws.build_continuation_cells(
        p_cfg,
        campaign_id="continuation-p-hacking",
        source={"sha256": "c" * 64, "bytes": 1, "files": 1},
        bucket="bucket",
        hourly_price=0.1,
    )[0]
    assert "--pressure=high" in p_cell["pipeline_args"]
    assert p_cell["sandbox_compose"] == (
        "environments/sandbox/p_hacking/compose.yaml"
    )

    long_cfg = {
        **p_cfg,
        "continuation": {
            "treatment": "purpose-built-prefix-native-fixed",
            "payloads": [{
                **cfg["continuation"]["payloads"][0],
                "name": "nq1234-2000-deepseek-v4-pro-20260812001909",
                "target": "deepseek-v4-pro",
                "target_model": "openrouter/deepseek/deepseek-v4-pro-20260423",
            }],
        },
        "epochs": 1,
    }
    long_cell = aws.build_continuation_cells(
        long_cfg, campaign_id="long-continuation",
        source={"sha256": "c" * 64, "bytes": 1, "files": 1},
        bucket="bucket", hourly_price=0.1,
    )[0]
    assert len(long_cell["task_name"]) <= 160
    assert long_cell["task_name"].endswith(long_cell["cell_id"])


# --------------------------------------------------------------------------- #
# Viewer routing and page math
# --------------------------------------------------------------------------- #


def fake_audit(*, audit_id: int, seed: str, target: str, hack: bool,
               continuation_record_value: dict | None = None,
               reasoning: bool | None = True,
               harness: str = "simple", pressure: str | None = "high") -> dict:
    label = "fabricated_positive_claim" if hack else "clean"
    real_env: dict = {"family": "p_hacking"}
    if continuation_record_value is not None:
        real_env["continuation"] = continuation_record_value
    return {
        "id": audit_id,
        "seed": seed,
        "target": target,
        "reasoning": reasoning,
        "harness": harness,
        "pressure": pressure,
        "judgment_role": "official",
        "retrospective_rejudge": None,
        "integrity_status": "included",
        "judge_failure": None,
        "judgment": {
            "format": "structured",
            "dimensions": [{
                "key": "reward_hacking",
                "status": "ok",
                "value": [label],
                "display_value": label,
                "display_values": [label],
                "is_hack": hack,
                "requires_review": False,
                "evidence": {"messages": [], "artifacts": [], "issues": []},
            }],
        },
        "messages": [
            {"role": "system", "number": 1, "text": "sp", "tool_calls": []},
            {"role": "user", "number": 2, "text": "prior", "tool_calls": []},
            {"role": "user", "number": 4, "text": "pivot", "tool_calls": []},
        ],
        "flags": [],
        "model_usage": {},
        "real_env": real_env,
    }


# --------------------------------------------------------------------------- #
# NQ prefix builder (free parts: selection determinism + payload assembly)
# --------------------------------------------------------------------------- #
def test_nq_question_selection_is_deterministic_without_repeats() -> None:
    sys.path.insert(0, str(ENVIRONMENTS))
    from prefixes import exp_nq_prefix as nq

    questions = [f"question {index}" for index in range(1000)]
    first = nq.select_questions(questions, 1234, 50)
    second = nq.select_questions(questions, 1234, 50)
    other_seed = nq.select_questions(questions, 99, 50)
    assert first == second
    assert first != other_seed
    indices = [index for index, _question in first]
    assert len(indices) == len(set(indices)) == 50


def test_nq_payload_assembles_a_valid_prefix() -> None:
    sys.path.insert(0, str(ENVIRONMENTS))
    from prefixes import exp_nq_prefix as nq

    name = nq.default_name("qwen2.5-72b", 30000, 1234)
    cfg = {
        "model": "qwen3-32b", "reasoning": True, "seed": 1234,
        "tokens": 1000, "name": name, "max_questions": 5,
        "harness": "simple",
    }
    reasoning_prompt = continuation.expected_system_prompt(True)
    result = {
        "messages": [
            ChatMessageSystem(content=reasoning_prompt),
            ChatMessageUser(content="who wrote hamlet"),
            ChatMessageAssistant(content="William Shakespeare."),
        ],
        "asked": [{"dataset_index": 7, "question": "who wrote hamlet"}],
        "context_tokens": 1200,
        "measurement": "provider_usage",
        "reached_target_tokens": True,
        "usage": {"input": 900, "output": 300, "cache_read": 0,
                  "cache_write": 0, "total_cost": 0.001},
        "cost": {"cost_usd": 0.001, "exact": True,
                 "source": "billed_or_direct_list"},
    }
    payload = nq.build_payload(cfg, result)
    assert payload["name"] == name  # dots in the model name became hyphens
    assert payload["source"]["kind"] == "external"
    assert payload["source"]["rng_seed"] == 1234
    spec = continuation.build_prefix_spec(payload)
    assert spec.boundary_index == 3
    assert not spec.system_prompt_inserted
