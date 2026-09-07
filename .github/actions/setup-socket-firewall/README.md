# Setup Socket Firewall wrapper

Composite GitHub Action that installs Socket Firewall Enterprise (`sfw`) in
wrapper mode and routes supported package-manager commands through it for later
bash steps in the same job.

## Usage

```yaml
- name: Setup Socket Firewall wrapper
  uses: ./.github/actions/setup-socket-firewall
  with:
    socket-api-key: ${{ secrets.SOCKET_API_KEY }}
```

Call the action after checkout and before the first `uv sync`, `uv lock`,
`uv pip`, or `pip install` step.

The action:

- downloads the Linux `sfw` binary when needed
- exports `SOCKET_API_KEY` and `SFW_TELEMETRY_DISABLED=true`
- configures `SFW_CUSTOM_REGISTRIES` with wrap hosts (including
  `files.pythonhosted.org` for PyPI artifact downloads)
- writes a `BASH_ENV` file with package-manager wrapper functions

Wrapper functions apply only to later bash steps that source `BASH_ENV`. They
do not affect third-party Actions or dependency installs inside Docker builds
(those use BuildKit secrets — see repository Dockerfiles).

## Fork PRs / missing secret

Graphiti is a public repository. Fork pull requests do not receive repository
secrets. When `socket-api-key` is empty, this action **soft-skips**: it prints a
notice, sets `SOCKET_FIREWALL_ENABLED=false`, and leaves package managers
unwrapped so CI still succeeds. Same-repo runs with `SOCKET_API_KEY` configured
get full enforcement.

## API key scopes

`SOCKET_API_KEY` should include the Socket scopes required for Enterprise
wrapper mode (`packages` and `entitlements:list`). Configure it as a repository
Actions secret and in the `development` / `release` environments used by CI.

## Docker builds

Official release workflows pass `socket_api_key` as a BuildKit secret so image
dependency fetches go through `sfw`. They also set the non-secret
`SOCKET_FIREWALL_ENABLED=true` build argument, which separates enforced release
layers from public fallback layers in the BuildKit cache. In enforced mode, a
missing or empty `socket_api_key` fails the build. Dockerfiles default the
argument to `false` and install directly so community `docker build` / Compose
usage continues to work without a Socket key.

### Cache bypass on enforced builds

`sfw` inspects network requests, so anything served from cache is never
scanned. Two caches would otherwise hide dependencies from an enforced release
build:

- the **BuildKit layer cache**, which skips the install step entirely when its
  inputs are unchanged
- the **`uv` cache mount**, which lets `uv` resolve wheels from disk without
  contacting PyPI even when the step does run

Enforced builds defeat both. Release workflows pass
`SOCKET_SCAN_ID=<run_id>-<run_attempt>`, which is unique per run and invalidates
the install layer, and the enforced branch exports `UV_NO_CACHE=1`. Without
these, an image whose dependencies had not changed could ship a package that
Socket flagged after the layer was first built. Community builds leave
`SOCKET_SCAN_ID` empty and keep both caches, so unauthenticated builds stay
fast.
