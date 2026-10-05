# Graphiti Configuration Templates

## Base Configuration

```json
{
  "group_id": "[project-name]",
  "description": "[project description]",
  "shared_group_ids": ["personal-prefs", "std-coding", "std-design", "std-arch", "kb-tech"],
  "shared_entity_types": ["Preference", "Convention", "DesignPattern", "Architecture", "Decision"],
  "shared_patterns": ["规范", "约定", "标准", "convention", "standard", "best practice"],
  "write_strategy": "llm_based"
}
```

## Development Projects

### Language-Specific Entity Types

| Language | Additional Entity Types |
| --- | --- |
| Rust | RustPattern, AsyncRuntime, Crate |
| Python | PythonPattern, DjangoApp, FlaskRoute |
| Go | GoPattern, Goroutine, APIEndpoint |
| Node.js | NPMPackage, NodePattern, ExpressRoute |
| Vue | VueComponent, VueComposable, PiniaStore |
| React | ReactComponent, ReactHook, ReduxAction |

### Language-Specific Patterns

| Language | Additional Patterns |
| --- | --- |
| Rust | "async", "tokio", "actix", "serde" |
| Python | "django", "fastapi", "pydantic", "asyncio" |
| Go | "goroutine", "channel", "defer", "interface" |
| Node.js | "express", "async/await", "npm", "typescript" |
| Vue | "composition api", "script setup", "pinia" |
| React | "hook", "useState", "useEffect", "context" |

## Learning Projects

### Entity Types

| Type | Description |
| --- | --- |
| Course | Course or tutorial |
| Concept | Key concept or知识点 |
| Note | Study notes |
| Question | Practice question |
| Progress | Learning progress |

### Patterns

- "课程", "course", "tutorial"
- "概念", "concept", "知识点"
- "笔记", "note", "记录"
- "练习", "exercise", "practice"
- "高考", "exam", "测试"

### Example

```json
{
  "group_id": "physics-learning",
  "description": "高中物理学习",
  "shared_group_ids": ["personal-prefs", "kb-tech"],
  "shared_entity_types": ["Course", "Concept", "Note", "Question", "Progress"],
  "shared_patterns": ["物理", "力学", "电磁学", "高考", "物理公式", "physics"],
  "write_strategy": "llm_based"
}
```

## Research Projects

### Entity Types

| Type | Description |
| --- | --- |
| Paper | Research paper |
| Experiment | Experiment setup |
| Data | Dataset |
| Finding | Research finding |
| Method | Research method |

### Patterns

- "论文", "paper", "research"
- "实验", "experiment", "数据"
- "方法", "method", "分析"
- "结论", "conclusion", "结果"

### Example

```json
{
  "group_id": "ml-research",
  "description": "机器学习研究",
  "shared_group_ids": ["personal-prefs", "kb-tech"],
  "shared_entity_types": ["Paper", "Experiment", "Data", "Finding", "Method"],
  "shared_patterns": ["机器学习", "论文", "实验", "ml", "research", "paper"],
  "write_strategy": "llm_based"
}
```

## General/Personal Knowledge

### Entity Types

| Type | Description |
| --- | --- |
| Idea | Ideas and thoughts |
| Reference | Reference material |
| Memory | Personal memory |
| Task | Tasks and todos |

### Patterns

- "想法", "idea", "thought"
- "参考", "reference", "资源"
- "记忆", "memory", "remember"
- "任务", "task", "todo"
