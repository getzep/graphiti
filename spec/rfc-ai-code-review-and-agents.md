# RFC: AI code review and coding agents in Graphiti

| Field | Value |
|---|---|
| Status | Draft |
| Author | Peter Evans |
| Date | 2026-09-25 |
| Scope | `getzep/graphiti` and its GitHub configuration |
| Related | `CONTRIBUTING.md`, `.github/workflows/pr-intake.yml`, `ellipsis.yaml` |

## Summary

Graphiti will use GitHub Copilot code review as the only AI reviewer. Copilot will review a pull request only when a maintainer requests the review. No Devin session and no other agent will execute Graphiti code on Zep infrastructure. When an agent must run code for Graphiti, the agent runs in an isolated sandbox that Zep does not operate.

## Motivation

Graphiti is a public repository with external contributors. A pull request from a fork is untrusted input. Two risks drive this RFC:

1. An AI reviewer that runs automatically on every pull request spends credits on spam and low-quality submissions, and gives contributors the impression that the review is the gate.
2. An agent that executes untrusted code on Zep-controlled infrastructure (a Devin VM, a developer laptop, or a Zep CI runner with credentials) exposes secrets and the internal network.

A survey of 35 high-star open-source repositories (internal research notes, September 2026) shows the common pattern: hosted review apps that read the diff only, and human-gated coding agents whose output a maintainer merges. The projects with the strongest posture (PyTorch, LangChain) run the agent in a sandbox that is separate from the review trigger.

## Decisions

### D1. Copilot is the AI reviewer

Copilot code review replaces Ellipsis. The `ellipsis.yaml` file is removed and the Ellipsis GitHub App is uninstalled from the repository.

Copilot review guidance is stored in `.github/copilot-instructions.md`. Copilot also reads `CLAUDE.md` and `AGENTS.md`, so those files stay accurate for review purposes. Path-specific instructions, when needed, are stored in `.github/instructions/**/*.instructions.md`.

### D2. Copilot reviews on request only

The repository does not enable the "Automatically request Copilot code review" branch rule. No organization or enterprise ruleset targets `graphiti` for automatic review.

A maintainer requests a review in one of two ways:

- In the pull request sidebar, under Reviewers, click Request next to Copilot.
- Through the REST API, request `copilot-pull-request-reviewer[bot]` as a reviewer.

Note: an `@copilot` mention in a comment starts the Copilot cloud agent (the coding agent), and does not start a code review. The review trigger is the reviewer request. `CONTRIBUTING.md` documents this distinction so that contributors do not expect a review from a mention.

Copilot leaves a "Comment" review. Copilot approvals do not count toward required approvals. A human maintainer approval remains the merge gate.

The default review effort level is Lite. A maintainer selects Balanced for a security-sensitive or cross-module change before the request.

### D3. Where Copilot review runs

Copilot code review executes its agentic steps (full project context gathering) on GitHub Actions runners. The repository uses standard GitHub-hosted runners for these runs. Copilot does not run Graphiti tests or build steps during a review; it reads the repository and the diff. The review compute is GitHub's, and the runner has no Zep secrets.

### D4. No agent executes Graphiti code on Zep infrastructure

Zep-operated agents (for example Devin sessions) are limited to read-only analysis of Graphiti and to authoring changes without executing them. Such an agent does not run `uv sync`, `pytest`, `docker`, or any repository script for Graphiti.

When a task needs execution (reproduce a bug, run integration tests, iterate on a fix), the work is delegated to an agent in an isolated sandbox. The sandbox has these properties:

- A separate kernel boundary (microVM or full VM) per run.
- No long-lived Zep credential inside the sandbox. AWS access, when needed, uses OIDC federation with `sts:AssumeRoleWithWebIdentity` to a role that grants only `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`.
- The coding agent runs inside the sandbox as the main process.
- Network egress limited to package registries, GitHub, and the Bedrock endpoint.

Providers that document both OIDC federation to AWS and an agent running inside are E2B, Modal, exe.dev, Cursor Cloud Agents, and Fly Machines. The provider choice is out of scope for this RFC and will be settled by a proof of concept on two providers.

### D5. Human-in-the-loop for agent-authored changes

An agent may open a pull request. Every agent-authored pull request follows the same path as a contributor pull request: the intake workflow labels it, a maintainer requests the Copilot review when useful, and a maintainer approves and merges. The pull request description states which agent produced the change. The existing `CONTRIBUTING.md` rule applies: AI assistance is acceptable and is not by itself a reason for `needs-rework`.

## Changes to the repository

| Item | Action |
|---|---|
| `ellipsis.yaml` | Delete |
| Ellipsis GitHub App | Uninstall from `getzep/graphiti` |
| `.github/copilot-instructions.md` | Add: review criteria (copyright header, single quotes, 100-character lines, no secrets or sensitive logging, `_int` suffix for integration tests, `@pytest.mark.integration`) |
| Branch rulesets | Verify that no ruleset enables automatic Copilot review for `graphiti` |
| `CONTRIBUTING.md` | Add a short "AI review and agents" section: Copilot reviews on maintainer request, `@copilot` does not trigger a review, agent-authored pull requests must disclose the agent |
| Copilot review effort | Set the repository default to Lite |

## Non-goals

- Selecting the sandbox provider.
- Enabling the Copilot cloud agent to author code in `graphiti`. A later RFC covers coding agents once the sandbox proof of concept is complete.
- Changing the intake automation in `.github/workflows/pr-intake.yml` and `issue-intake.yml`.

## Open questions

1. Should the Copilot review request be limited to maintainers with write access, or should a label such as `ai-review` applied by a maintainer trigger the request through a workflow? A label-driven workflow gives an audit trail but adds a `pull_request_target` job with `pull-requests: write`.
2. Should the `ai-moderator.yml` workflow's `ai-generated` label continue to be applied now that AI assistance is explicitly acceptable?
3. Does `CLAUDE.md` stay as a Copilot-readable instruction file, or is its content merged into `AGENTS.md` to keep one source?

## References

- GitHub Docs, Using GitHub Copilot code review: https://docs.github.com/en/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review
- GitHub Docs, Configuring automatic code review: https://docs.github.com/en/copilot/how-tos/use-copilot-agents/request-a-code-review/configure-automatic-review
- GitHub Docs, About Copilot code review (agentic capabilities and runners): https://docs.github.com/en/copilot/concepts/agents/code-review
- PyTorch hardened review workflows: https://github.com/pytorch/pytorch/blob/main/.github/workflows/hardened-pr-review.yml
- LangChain Open SWE: https://github.com/langchain-ai/open-swe
