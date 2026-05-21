---
name: feedback-prefs
description: "User's working preferences and things to avoid — no git push without permission, no host installs, concise responses"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e569bc8-a2d7-4a62-98c5-40ce17bb07d3
---

**No git push without explicit instruction.**
**Why:** User said "do git pushes only when I say so."
**How to apply:** Never run `git push` unless the user explicitly asks in that message.

**Don't install packages on the host.**
**Why:** User said "what are you installing? don't break anything" when pip was run on host.
**How to apply:** All pip/package installs go inside the container via `docker exec`. Never run pip on the host shell.

**Run commands inside the container, not the host.**
**Why:** Isaac-sim, isaacteleop, and all related Python packages live inside `isaac-lab-base` container. System Python on host is 3.10 and lacks all packages.
**How to apply:** Use `docker exec isaac-lab-base bash -c "..."` for any sim/teleop-related commands. `./isaaclab.sh` from `/root/groot` on the host resolves to system Python 3.10 and fails.

**Keep responses concise.**
**Why:** User reads diffs directly, doesn't need trailing summaries.
**How to apply:** No recap at end of response. State what changed, what's next — one or two sentences max.

**Ask before running broad exploratory commands.**
**Why:** User interrupted multiple broad `find /` searches.
**How to apply:** Use targeted paths (known install locations) rather than scanning the entire filesystem.
