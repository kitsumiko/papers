# Native Darwin execution

Operator directive, 2026-11-17: every build, test and toolchain runs inside a native Darwin pod. This applies equally
to Claude and Codex and supersedes older host-build or host-install exceptions in examples, skills and runbooks.

- Run every compiler, build, test, linter and renderer in the project's approved native Darwin k3sm pod or native
  dollhouse. This includes Swift/Xcode, Go, C/C++, Rust, Python and Node tooling. No host compiler or local fallback;
  no VM, Multipass or Docker Desktop fallback for development tooling.
- Keep the project's existing `yurei build` queue and gate ownership. No GitHub Actions or other hosted CI runners.
  A queued job alone does not prove placement: verify
  the native executor, namespace/pod and source revision, and retain actual execution evidence. Commands shown in
  this repository describe work inside that route; copied examples never authorize host execution.
- Host `git`, `gh`, `kubectl`/`k3sm` and verified source/data-transfer orchestration remain available. Use advertised
  scoped tools or the runtime's executable absolute `YUREI_BIN`; never assume bare `yurei` is on PATH or invent a
  scoped tool. A transport wrapper must keep every compiler and test inside the pod.
- Never install software on the host Mac. No brew/port installs or upgrades or global language-package installs.
  No downloaded or piped installers, sudo installs, application-folder writes or LaunchAgents.
  Existing local toolchains, virtual environments, lab installers and app-relaunch recipes do not create an exception.
- A missing tool, unsupported toolchain, unavailable pod or routine shell/redirect/parity problem is assigned
  internal repair with an owner and next step. Preserve an honest blocked status until the supported route works;
  never ask for an operator parity decision, install on the host, bypass a refusal or silently widen permissions.
- Preserve independent review, held-out checks, supply-chain pins, release restrictions, HALT/run controls, leases
  and every canonical ledger/config writer. Credentials, money beyond an approved envelope, legal/external
  commitments and irreversible operations retain their real human gates. This rule grants none of those actions.
