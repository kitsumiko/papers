# Agent rules

Rules for any agent or console working in this repo. They apply to every project the operator owns.

## Containers only (supply-chain safety)

Operator, 2026-11-16, verbatim: "please make it a rule across EVERY repo that we are to run things in containers on
purpose because of supply chain issues". And: "no console is allowed to install things on the host machine, and can
only run things in containers".

- Never install software on the host Mac: no `brew`/`port` install, upgrade or tap; no global `pip`/`pipx`/`npm -g`/
  `pnpm -g`/`yarn global`/`cargo install`/`go install`/`gem install`; no `curl … | sh` installers or downloaded
  `.pkg`/`.dmg` installers; no writes to `/Applications`; no `sudo` installs.
- Toolchains and services run in containers: k3sm pods (native Darwin arm64). A tool that is not in a container image
  yet is added to one, pinned to an exact version and sha256-verified, or you stop and ask the operator.
- Allowed on the host: `git`, `gh`, `kubectl` and the k3sm CLI, and the `yurei` CLI that ships with the Yureisha app.

## No CI on GitHub

Operator, 2026-11-16, verbatim: "we should not be building anything with CI runners ever, make that a rule please,
it's too insecure". And: "ci in github* nothing should ever be there in any project". And: "we should be running ci
locally, using yureisha".

No CI on GitHub: no GitHub Actions workflows, ever, in any project, and Actions stays disabled; all gates run locally
in k3sm pods.

- Never add a file under `.github/workflows/`, and never build, test or release on a hosted runner (GitHub Actions or
  any other hosted CI).
- GitHub Actions is disabled in this repo's settings. Never turn it back on.
- Gates, if this repo has any, run locally through Yureisha CI (yureisha/ci status) or the repo's own gate script, in
  a k3sm pod.
