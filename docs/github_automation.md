# GitHub automation

Pull requests use conventional titles, automatic labels, and squash-only
merges. The repository protects `main`, including administrator merges, with
up-to-date branches, resolved conversations, and required CI checks.

## Pull requests and dependency updates

Use a title such as `fix: preserve episode identities`, `feat: add an adapter`,
or `feat!: remove an old contract`. The PR title becomes the squash commit
title, so a series of development commits produces one release entry.

The PR bot maintains `area:*`, `size:*`, and `release:*` labels. It removes
outdated generated labels after changes, without removing human controls:

- `automerge` opts a PR into automatic squash merging.
- `hold` blocks automated merging, including otherwise eligible dependency PRs.
- `automerge:dependencies` is set only for verified Dependabot patch/minor
  updates without maintainer changes. Major updates require explicit opt-in.
- Draft PRs never merge automatically. Release PRs require review or explicit
  `automerge` opt-in, just like other non-Dependabot PRs.

Dependabot checks GitHub Actions and `uv` dependencies weekly. Patch/minor
updates are grouped; major updates remain separate. The validated `torch`,
`torchvision`, and `mani-skill` pins are excluded from automatic version updates
because they must be evaluated together on compatible simulator infrastructure.

## CI and merge protection

Required checks are `test`, `package`, `installed-artifact (3.10)`,
`installed-artifact (3.13)`, and `CI gate`, supplied by GitHub Actions.
`CI gate` requires the core jobs, conventional-title/path checks, automation
tests, and Actionlint to succeed. Container-related changes also require all
three existing container smoke profiles through a reusable workflow. A skipped
or cancelled required job cannot make the gate green.

The bot also inspects additional reported checks and external commit statuses.
Missing, pending, failed, cancelled, or truncated check results block merging.
Intentional skips are limited to inapplicable container jobs and the optional
self-hosted ManiSkill GPU job. No review or branch-protection bypass is used.
The merge request includes the exact checked head SHA and rechecks the PR's
draft state and labels immediately beforehand.

Privileged PR automation checks out only `main`; it never executes PR code or
downloads PR artifacts. PR titles and bodies are passed to GitHub APIs, not
interpolated into shell commands. The normal CI workflow has read-only access.

## Versions and publication

After successful CI on the current `main` commit, Release Please creates or
updates a version PR. `fix`, `perf`, and `revert` produce a patch bump; `feat`
produces a minor bump; a `!` or `BREAKING CHANGE:` footer produces a major bump.
Documentation, CI, and maintenance changes alone do not require a release.
Major bumps are honored even before version 1.0.

`nyssa_bench/version.py` remains the package-version source. The release bot
updates it, `CHANGELOG.md`, and `.release-please-manifest.json` together; the
version validator rejects manifest drift. The bootstrap SHA starts automated
release history after the already-recorded 0.1.0 development work.

Bot-created version PRs receive explicit CI and labeling dispatches. When a
version PR is merged and main CI passes, Release Please creates its version tag
and GitHub release, then dispatches the existing package/container publication
workflow at that tag. The release workflow attaches its immutable artifacts to
the release and preserves the existing PyPI/TestPyPI environment approvals.

The workflow explicitly dispatches CI after bot merges and for release PRs,
and dispatches publication after bot-created tags. This avoids relying on
downstream events suppressed or approval-gated by `GITHUB_TOKEN`.
No personal access token or additional repository secret is required.

## Administration and recovery

Repository settings must keep merge commits and rebase merges disabled, squash
and auto-merge enabled, and squash titles set to the PR title. Actions needs
permission to create PRs; individual workflows request only their required
permissions. Keep the five required checks above strict and pinned to the
GitHub Actions app. Existing review and publishing protections stay in force.

Use **PR automation → Run workflow** with a PR number to reconcile labels and
retry merging after a resolved review or external check; use `0` to synchronize
the label catalog only. Completed CI runs and a 15-minute reconciliation schedule
also retry eligible PRs after reviews, conversations, or external checks change. If a
branch falls behind main, update it and allow CI to rerun.

Use **Release PR → Run workflow** on `main` to reconcile release PRs after
main's CI has passed. Use **Release Python package → Run workflow** at a release
tag if publication was not dispatched. Publication retains its existing refusal
to overwrite immutable container version tags.

Local validation:

```bash
node --test scripts/github/*.test.cjs
actionlint
uv run pre-commit run --all-files
uv run pre-commit run --hook-stage pre-push --all-files
```

References: [GitHub workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow),
[Release Please](https://github.com/googleapis/release-please-action), and
[Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
