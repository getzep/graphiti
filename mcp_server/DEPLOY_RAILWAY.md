# Deploying Graphiti MCP Server on Railway

This guide captures the minimum configuration needed to deploy the Graphiti MCP server on Railway without encountering the `cd: mcp_server: No such file or directory` error.

## Root cause

Railway's "Root Directory" setting was already pointed at `mcp_server/`. The start command attempted to run `cd mcp_server && ...`, so the service tried to change into `mcp_server/mcp_server/` at runtime and failed, resulting in a 502 error.

## Recommended configuration (Root Directory = `mcp_server`)

1. In Railway → **Settings → Root Directory**, set the value to `mcp_server`.
2. Set the **Start Command** to:

   ```bash
   bash start.sh
   ```

Railway will execute the launcher inside the `mcp_server` directory, respect the `$PORT` value it injects, and the server will bind to `0.0.0.0:$PORT`.

## Alternative configuration (Root Directory = repo root `/`)

If you prefer to keep the root directory at the repository root, update the start command instead:

```bash
cd mcp_server && bash start.sh
```

This mirrors the local workflow while continuing to honor the `$PORT` value supplied by Railway.

## Local verification

Install dependencies with `uv sync`, then run:

```bash
cd mcp_server
PORT=8080 bash start.sh
```

You should see the log line:

```
Graphiti MCP Server listening on 0.0.0.0:8080 (transport=sse)
```

## Remote validation with MCP Inspector

After deploying to Railway, validate the endpoint with [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector \
  --transport sse \
  --url https://<railway-app>.railway.app/sse \
  --headers "Authorization: Bearer <KEY-if-configured>"
```

Replace `<railway-app>` with your Railway subdomain and include the authorization header only if your deployment requires it.
