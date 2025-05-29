### Graphiti MCP Memory Guide — Compact Edition

**1. Before each task**

- `search_nodes` → filter by **Preference | Procedure | Requirement**
- `search_facts` → grab related relationships
- Review every match before working

**2. Add / update knowledge**

- On any new or changed requirement / preference → `add_episode` (split long items)
- Flag updates clearly; record only what’s new
- Save discovered **procedures** & **facts** with precise categories

**3. While working**

- Honor preferences, follow procedures exactly, apply facts
- Keep outputs consistent with stored knowledge

**4. Best practices**

- Always search first
- Combine node + fact queries for complex tasks
- Use `center_node_uuid` to explore related info
- Prefer specific matches over general ones
- Log repeating user patterns as prefs/procs

---

```mermaid
flowchart TD
    A[Start Task] --> B[search_nodes<br/>(Pref/Proc/Req)]
    B --> C[search_facts]
    C --> D[Review matches]
    D --> E{New info?}
    E -- Yes --> F[add_episode]
    F --> G[Update<br/>procedures & facts]
    G --> H[Do work<br/>respect prefs/procs/facts]
    E -- No --> H
```

Use the graph as your memory—search first, add promptly, act consistently.
