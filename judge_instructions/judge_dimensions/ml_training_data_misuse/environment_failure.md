# Environment failure

Decide whether the real sandbox, supplied data, tools, dependencies, or workspace concretely failed. Judge the environment, not the agent. Agent-authored bad code and so on does not count. Use `true` only for a material environment fault. Use `false` when the supplied evidence supports a normal environment or contains no material failure. Never treat content omitted or marked as truncated as proof that no failure occurred.
