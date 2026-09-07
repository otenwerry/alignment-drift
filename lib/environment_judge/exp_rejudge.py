"""Inspect scorer that applies the current judge to a stored trajectory."""

from __future__ import annotations

from inspect_ai.model import get_model
from inspect_ai.scorer import NOANSWER, Score, Target, scorer
from inspect_ai.solver import TaskState

from .exp_real import judge_complete_real_trajectory
from .exp_runner import JudgeCallError, JudgePostValidationError
from .prompt import JUDGE_SCHEMA_VERSION
from .score import score_from_judgment


@scorer(metrics=[], name="environment_judge")
def retrospective_rejudge():
    async def score(state: TaskState, target: Target) -> Score:
        del target
        metadata = state.metadata or {}
        source = metadata.get("retrospective_rejudge") or {}
        real_env = metadata.get("real_env") or {}
        family = source.get("family")
        try:
            result = await judge_complete_real_trajectory(
                family=family,
                messages=list(state.messages),
                real_env=real_env,
                model=get_model(role="judge"),
            )
        except JudgePostValidationError as error:
            return Score(
                value=NOANSWER,
                explanation=str(error),
                metadata={
                    "environment_judge": {
                        **error.call_metadata,
                        "judgment_role": "retrospective_rejudge",
                        "retrospective_rejudge": source,
                    }
                },
            )
        except JudgeCallError as error:
            return Score(
                value=NOANSWER,
                explanation=str(error),
                metadata={
                    "environment_judge": {
                        **error.call_metadata,
                        "judgment_role": "retrospective_rejudge",
                        "retrospective_rejudge": source,
                    }
                },
            )
        except Exception as error:
            return Score(
                value=NOANSWER,
                explanation=repr(error)[:500],
                metadata={
                    "environment_judge": {
                        "judge_schema_version": JUDGE_SCHEMA_VERSION,
                        "family": family,
                        "official_stage": "final",
                        "post_validation": "call_failed",
                        "judgment_role": "retrospective_rejudge",
                        "retrospective_rejudge": source,
                    }
                },
            )
        if not isinstance(result.value, dict):
            envelope = dict((result.metadata or {}).get("environment_judge") or {})
            envelope.update({
                "judgment_role": "retrospective_rejudge",
                "retrospective_rejudge": source,
            })
            return Score(
                value=NOANSWER,
                explanation=result.explanation or "judge returned no structured answer",
                metadata={"environment_judge": envelope},
            )
        return score_from_judgment(
            judgment=result.value,
            call_metadata=dict(result.metadata or {}),
            family=family,
            official_stage="final",
            reused_stage_one=False,
            extra_envelope={
                "judgment_role": "retrospective_rejudge",
                "retrospective_rejudge": source,
            },
        )

    return score
