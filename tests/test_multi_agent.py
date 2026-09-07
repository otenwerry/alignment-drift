"""Free tests for fresh-session ACTIVITY_LOG.md treatments."""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ContentToolUse,
    get_model,
)
from inspect_ai.tool import ToolCall


ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from multi_agent_evidence import (  # noqa: E402
    ACTIVITY_LOG_PROMPT_TEMPLATE,
    caveat_fields,
    effective_opening_user_message,
    finalize_activity_log_exposure,
    judge_view_messages,
    render_activity_log,
    sanitize_artifacts,
    update_activity_log_exposure,
    update_activity_log_exposure_from_model_events,
)
from real_integrity import sample_integrity_record  # noqa: E402
from exp_aws_trajectory import build_multi_agent_cells  # noqa: E402
from exp_real_audit import (  # noqa: E402
    _finalize_real_sample,
    assemble_real_protocol,
    build_real_task,
)
from exp_real_continuation import expected_system_prompt  # noqa: E402
from multi_agent_evidence import ActivityLogRun  # noqa: E402
from project_paths import SEEDS_ROOT  # noqa: E402
import environment_judge.exp_real as judge_exp_real  # noqa: E402


def test_pipeline_requires_explicit_harness(monkeypatch) -> None:
    sys.path.insert(0, str(ENVIRONMENTS))
    import exp_multi_agent_pipeline as pipeline

    monkeypatch.setattr(sys, "argv", [
        "exp_multi_agent_pipeline.py",
        "--treatment=prior-hack",
        "--activity-logs=1",
        "--seed-dir=p_hacking",
        "--seeds=checkout_redesign",
        "--epochs=1",
    ])
    try:
        pipeline._parse_args()
    except SystemExit as error:
        assert "--harness is required" in str(error)
    else:
        raise AssertionError("missing --harness was accepted")


def test_aws_cell_ships_activity_payload_to_fresh_session_pipeline() -> None:
    cfg = {
        "multi_agent": {
            "treatment": "prior-hack",
            "payloads": [{
                "name": "log-traj1",
                "sha256": "a" * 64,
                "file_sha256": "b" * 64,
                "local_path": "/tmp/activity.json",
                "target": "qwen3-32b",
                "target_model": "openrouter/qwen/qwen3-32b",
                "reasoning": False,
            }],
        },
        "seeds": ["checkout_redesign"],
        "epochs": 1,
        "seed_dir": "p_hacking",
        "seeds_path": "/seeds/p_hacking",
        "family": "p_hacking",
        "harness": "simple",
        "condition": "allow",
        "judge_resolved": "openai/gpt-5.6-luna",
        "gate_model": "openai/gpt-5.6-luna",
        "time_limit": 1800,
        "pressure": "low",
        "aws_region": "us-west-2",
        "aws_instance_type": "c7a.xlarge",
        "worker_allowed_secret_names": [],
    }
    with patch(
        "exp_aws_trajectory._sandbox_compose_for_target",
        return_value="environments/sandbox/p_hacking/compose.yaml",
    ):
        cells = build_multi_agent_cells(
            cfg,
            campaign_id="campaign",
            source={"sha256": "c" * 64, "bytes": 123},
            bucket="bucket",
            hourly_price=0.2,
        )
    assert len(cells) == 1
    cell = cells[0]
    assert cell["pipeline_script"] == "exp_multi_agent_pipeline.py"
    assert cell["activity_log_name"] == "log-traj1"
    assert cell["activity_log_payload_sha256"] == "b" * 64
    assert any(
        argument.startswith("--activity-log-files=/var/lib/mats-worker/")
        for argument in cell["pipeline_args"]
    )


def test_shared_task_builder_mounts_log_and_changes_only_user_opening(
    tmp_path,
) -> None:
    record = _record()
    unit_path = SEEDS_ROOT / "p_hacking" / "checkout_redesign"
    system_prompt = expected_system_prompt(False)
    base_prompt = assemble_real_protocol(
        unit_path, "allow", system_prompt, "low"
    ).opening_user_message
    record.update({
        "version": "environments-multi-agent-v2",
        "name": "test-log",
        "treatment": "prior-hack",
        "source": {"kind": "external", "harness": "simple"},
        "base_opening_user_message": base_prompt,
        "effective_opening_user_message": effective_opening_user_message(base_prompt),
    })
    markdown = tmp_path / "ACTIVITY_LOG.md"
    markdown.write_text(record["activity_log"]["content"])
    mock_model = get_model("mockllm/model")
    target_build = SimpleNamespace(
        routed_slug="openrouter/qwen/qwen3-32b",
        model=mock_model,
        reasoning_enabled=False,
        strict_tools=False,
        prompt_cache_key=None,
    )
    task = build_real_task(
        target_name="qwen3-32b",
        target_build=target_build,
        unit_path=unit_path,
        unit_sp=system_prompt,
        reasoning=False,
        condition="allow",
        judge_model="mockllm/model",
        gate_model="mockllm/model",
        run_label="test",
        activity_log=ActivityLogRun(content_path=markdown, record=record),
        harness="simple",
        pressure="low",
    )
    sample = task.dataset[0]
    assert sample.input == effective_opening_user_message(
        record["base_opening_user_message"]
    )
    assert sample.files["/workspace/ACTIVITY_LOG.md"] == str(markdown)
    assert sample.metadata["protocol"]["system_prompt"] == system_prompt
    assert sample.metadata["protocol"]["opening_user_message"] == sample.input


def _source_messages():
    return [
        ChatMessageSystem(content="SOURCE SYSTEM MUST NOT BE COPIED"),
        ChatMessageUser(content="Analyze the prior task."),
        ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning="PRIVATE REASONING MUST NOT BE COPIED"),
                ContentText(text="I will inspect it."),
            ],
            tool_calls=[ToolCall(
                id="source-call", function="bash", arguments={"cmd": "ls"}
            )],
        ),
        ChatMessageTool(
            content="data.csv\nRESULT.md",
            tool_call_id="source-call",
            function="bash",
        ),
        ChatMessageAssistant(content="The prior analysis is complete."),
    ]


def _record() -> dict:
    content, metadata = render_activity_log(_source_messages())
    return {
        "activity_log": {"content": content, "metadata": metadata},
        "effective_opening_user_message": effective_opening_user_message("New task"),
    }


def test_activity_log_is_complete_observable_record_with_exact_prompt() -> None:
    record = _record()
    content = record["activity_log"]["content"]
    metadata = record["activity_log"]["metadata"]
    assert "SOURCE SYSTEM MUST NOT BE COPIED" not in content
    assert "PRIVATE REASONING MUST NOT BE COPIED" not in content
    for visible in (
        "Analyze the prior task.",
        "I will inspect it.",
        '"cmd": "ls"',
        "data.csv",
        "The prior analysis is complete.",
    ):
        assert visible in content
    assert metadata["omitted_system_message_count"] == 1
    assert metadata["omitted_reasoning_block_count"] == 1
    assert all(
        line.startswith(f"[L{number:06d}]")
        for number, line in enumerate(content.splitlines(), start=1)
    )
    assert effective_opening_user_message("New task") == (
        ACTIVITY_LOG_PROMPT_TEMPLATE.replace("{prompt}", "New task")
    )


def test_activity_log_keeps_embedded_tool_calls_and_results() -> None:
    content, metadata = render_activity_log([
        ChatMessageAssistant(content=[
            ContentText(text="I will inspect it."),
            ContentToolUse(
                id="embedded-call",
                tool_type="code_execution",
                name="bash",
                arguments="cat /workspace/RESULT.md",
                result="embedded tool result",
            ),
        ]),
    ])
    assert "cat /workspace/RESULT.md" in content
    assert "embedded tool result" in content
    assert metadata["included_tool_call_count"] == 1
    assert metadata["included_tool_result_count"] == 1


def test_exposure_requires_every_exact_numbered_line_in_model_input() -> None:
    record = _record()
    content = record["activity_log"]["content"]
    lines = content.splitlines()
    call = ChatMessageAssistant(
        content="Reading it.",
        tool_calls=[ToolCall(
            id="read-log",
            function="bash",
            arguments={"cmd": "cat /workspace/ACTIVITY_LOG.md"},
        )],
    )
    partial = ChatMessageTool(
        content="\n".join(lines[:-1]),
        tool_call_id="read-log",
        function="bash",
    )
    exposure = update_activity_log_exposure(record, [call, partial])
    assert exposure["status"] == "partial"
    assert exposure["fully_delivered"] is False
    assert exposure["missing_line_numbers"] == [len(lines)]

    final_chunk = ChatMessageTool(
        content=lines[-1],
        tool_call_id="read-log-2",
        function="bash",
    )
    exposure = update_activity_log_exposure(record, [final_chunk])
    assert exposure["status"] == "full"
    assert exposure["fully_delivered"] is True
    assert exposure["covered_line_count"] == len(lines)


def test_native_loss_makes_unproven_exposure_unknown() -> None:
    record = _record()
    exposure = finalize_activity_log_exposure(
        record,
        native_loss_events=[{"kind": "context_compaction"}],
    )
    assert exposure["status"] == "unknown"
    assert exposure["fully_delivered"] is None


def test_production_exposure_uses_each_successful_model_event_input() -> None:
    record = _record()
    lines = record["activity_log"]["content"].splitlines()
    midpoint = len(lines) // 2

    def event(content, **overrides):
        values = {
            "event": "model",
            "pending": False,
            "error": None,
            "output": SimpleNamespace(),
            "input": [ChatMessageTool(
                content=content,
                tool_call_id="read-log",
                function="bash",
            )],
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    checked = update_activity_log_exposure_from_model_events(record, [
        event("\n".join(lines), error="provider error"),
        event("\n".join(lines[:midpoint])),
        event("\n".join(lines[midpoint:])),
        event("\n".join(lines), pending=True),
    ])
    exposure = record["exposure"]
    assert checked == 2
    assert exposure["status"] == "full"
    assert exposure["successful_model_inputs_checked"] == 2
    assert exposure["measurement_sources"] == [
        "transcript_model_event_input"
    ]


def test_embedded_activity_log_read_is_measured_and_narrowly_masked() -> None:
    record = _record()
    content = record["activity_log"]["content"]
    message = ChatMessageAssistant(content=[
        ContentText(text="I read the previous agent's log."),
        ContentToolUse(
            id="embedded-read",
            tool_type="code_execution",
            name="bash",
            arguments="cat /workspace/ACTIVITY_LOG.md",
            result=f"{content}CURRENT OUTPUT\n",
        ),
    ])

    exposure = update_activity_log_exposure(record, [message])
    judged, masking = judge_view_messages([message], record)
    tool_use = next(
        block for block in judged[0].content
        if isinstance(block, ContentToolUse)
    )

    assert exposure["status"] == "full"
    assert content not in tool_use.result
    assert "CURRENT OUTPUT" in tool_use.result
    assert masking["tool_result_message_numbers_masked"] == [1]


def test_prior_log_content_is_mechanically_removed_from_judge_evidence() -> None:
    record = _record()
    content = record["activity_log"]["content"]
    call = ChatMessageAssistant(
        content="I read the attached log.",
        tool_calls=[ToolCall(
            id="read-log",
            function="bash",
            arguments={"cmd": "cat ACTIVITY_LOG.md"},
        )],
    )
    result = ChatMessageTool(
        content=content,
        tool_call_id="read-log",
        function="bash",
    )
    own_statement = ChatMessageAssistant(content="Now I will do the new task.")
    judged, masking = judge_view_messages([call, result, own_statement], record)
    assert content not in "\n".join(message.text or "" for message in judged)
    assert "omitted from judge evidence" in judged[1].text
    assert judged[0].text == call.text
    assert judged[2].text == own_statement.text
    assert masking["tool_result_message_numbers_masked"] == [2]
    assert masking["source_line_count_masked"] == len(content.splitlines())
    assert caveat_fields({"judge_masking": masking})["code"] == (
        "prior_agent_activity_log_omitted"
    )

    mixed = ChatMessageTool(
        content=f"{content.splitlines()[0]}\nCURRENT TEST OUTPUT\n",
        tool_call_id="read-log",
        function="bash",
    )
    mixed_judged, _ = judge_view_messages([call, mixed], record)
    assert content.splitlines()[0] not in mixed_judged[1].text
    assert "CURRENT TEST OUTPUT" in mixed_judged[1].text

    marker_stripped = ChatMessageTool(
        content="The prior analysis is complete.",
        tool_call_id="unlinked-copy",
        function="bash",
    )
    stripped_judged, _ = judge_view_messages([marker_stripped], record)
    assert stripped_judged[0].text == marker_stripped.text

    artifacts, paths = sanitize_artifacts({
        "files": [
            {"path": "/workspace/RESULT.md", "content": content},
            {"path": "/workspace/ACTIVITY_LOG.md", "content": content},
        ],
    }, record)
    assert paths == ["/workspace/ACTIVITY_LOG.md"]
    assert artifacts["files"][0]["content"] == content
    assert content not in artifacts["files"][1]["content"]


def test_shared_final_judge_path_applies_activity_log_mask(monkeypatch) -> None:
    record = _record()
    content = record["activity_log"]["content"]
    messages = [
        ChatMessageUser(content="New task"),
        ChatMessageAssistant(
            content="Reading prior activity.",
            tool_calls=[ToolCall(
                id="read-log",
                function="bash",
                arguments={"cmd": "cat ACTIVITY_LOG.md"},
            )],
        ),
        ChatMessageTool(
            content=content,
            tool_call_id="read-log",
            function="bash",
        ),
        ChatMessageAssistant(content="My current-task answer."),
    ]
    seen = {}

    async def fake_judge_trajectory(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(value={}, metadata={})

    monkeypatch.setattr(judge_exp_real, "judge_trajectory", fake_judge_trajectory)
    asyncio.run(judge_exp_real.judge_complete_real_trajectory(
        family="p_hacking",
        messages=messages,
        real_env={
            "multi_agent": record,
            "artifacts": {
                "files": [{
                    "path": "/workspace/RESULT.md",
                    "content": content,
                    "truncated": False,
                }],
            },
        },
        model=None,
    ))
    judged = "\n".join(message.text or "" for message in seen["messages"])
    assert content not in judged
    assert "My current-task answer." in judged
    assert "prior_agent_activity_log_omitted" in {
        caveat.code for caveat in seen["upstream_caveats"]
    }
    result_artifact = next(
        artifact for artifact in seen["artifacts"]
        if artifact.path == "/workspace/RESULT.md"
    )
    assert result_artifact.content == content.rstrip("\n")


def test_finalization_stores_and_grades_raw_multi_agent_artifacts() -> None:
    record = _record()
    copied_log = record["activity_log"]["content"]
    raw_artifacts = {
        "files": [{
            "path": "/workspace/predictions.csv",
            "content": copied_log,
            "truncated": False,
        }],
        "models": [],
        "lossy_reasons": [],
    }
    real_env = {
        "multi_agent": record,
        "harness": {"mode": "simple"},
        "judge_evidence_policy": {"lossy_reasons": []},
    }
    spec = SimpleNamespace(
        family="ml_training_data_misuse",
        deliverables=("predictions.csv",),
        gate=SimpleNamespace(),
    )
    state = SimpleNamespace(sample_id="sample", epoch=1)

    with (
        patch(
            "exp_real_audit.workspace_snapshot",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "exp_real_audit.extract_artifacts",
            new=AsyncMock(return_value=raw_artifacts),
        ),
        patch(
            "exp_real_audit.grade_predictions",
            return_value={"scored": True},
        ) as grade,
    ):
        asyncio.run(_finalize_real_sample(
            state,
            spec,
            real_env,
            {},
            None,
            target_name="qwen3-32b",
            target_slug="mock/model",
            reasoning=False,
            native_session_id=None,
        ))

    assert real_env["artifacts"]["files"][0]["content"] == copied_log
    assert grade.call_args.args[1] == copied_log


def test_incomplete_exposure_is_a_stored_integrity_exclusion() -> None:
    record = _record()
    finalize_activity_log_exposure(record)
    sample = SimpleNamespace(
        id="new-task",
        epoch=1,
        metadata={
            "real_env": {
                "multi_agent": record,
                "protocol": {"ended_reason": "protocol_end"},
                "artifacts": {"files": []},
                "harness": {"mode": "simple"},
                "gates": [],
            }
        },
        role_usage={"target": SimpleNamespace(output_tokens=1)},
        messages=[ChatMessageAssistant(content="Done")],
        events=[],
        scores={
            "environment_judge": SimpleNamespace(metadata={
                "environment_judge": {
                    "result": {
                        "environment_failure": {"value": False},
                    },
                    "post_validation": "passed",
                }
            })
        },
    )
    integrity = sample_integrity_record("task", sample, "model")
    assert integrity["status"] == "excluded"
    assert "activity_log_not_fully_delivered" in integrity["issues"]
