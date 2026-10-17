# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| Latest `0.x` release | Yes |
| Older releases | No |

## Scope

This policy covers the `graphiti-core` library.

The MCP server (`mcp_server/`) and the REST server (`server/`) are reference implementations. These servers show how to put Graphiti behind a protocol. These servers are not hardened for production use:

- The servers have no authentication and no authorization.
- The servers do not isolate callers from each other. Any caller can read, change, or delete data in any `group_id`.
- The example Docker Compose files use default database credentials.

Do not expose these servers to an untrusted network. If you deploy a server, put it behind your own authentication, authorization, and network controls.

Graphiti gives the same trust to all episodes in a `group_id`. A newer episode can change or invalidate facts from an older episode. The application must decide which content to ingest and into which `group_id`.

These reports are out of scope:

- Missing authentication, missing authorization, or default credentials in the reference servers.
- Untrusted episode content that adds, changes, or invalidates facts in the same `group_id`.

## Reporting a Vulnerability

Do **not** report suspected vulnerabilities in a public GitHub issue, pull request, discussion,
or community chat.

Use GitHub's private vulnerability reporting flow:

**[Report a vulnerability privately](https://github.com/getzep/graphiti/security/advisories/new)**

Include:

- A description of the vulnerability and its potential impact
- The affected Graphiti version, component, and configuration
- Reproduction steps or a proof of concept
- Any known mitigations or workarounds
- Whether the vulnerability has been disclosed elsewhere

Remove real credentials, customer data, and other sensitive information from the report.

Maintainers will acknowledge the report as soon as practical, assess its impact, and coordinate
remediation and disclosure with the reporter. Please allow maintainers time to investigate and
release a fix before sharing details publicly.
