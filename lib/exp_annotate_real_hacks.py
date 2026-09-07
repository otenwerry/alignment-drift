"""Retired compatibility entry for the former paid hack-turn annotator.

The structured judge now records per-dimension message and artifact evidence directly,
so a second model pass for turn localization would duplicate the official record.
"""


async def annotate_real_hacks(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "hack-turn annotation was retired; use environment_judge evidence references"
    )
