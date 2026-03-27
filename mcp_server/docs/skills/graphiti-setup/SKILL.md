---
name: graphiti-setup
description: |
  Graphiti Memory MCP configuration manager for any project type.
  Use when: (1) Setting up memory for new projects or learning topics, (2) Generating .graphiti.json configurations,
  (3) Detecting project type and generating appropriate memory settings, or (4) Managing memory patterns across different domains
---

# Graphiti Setup

## Overview

Manages Graphiti Memory MCP configuration for any project type. Detects project characteristics and generates appropriate `.graphiti.json` files to enable persistent memory across sessions.

## Quick Start

1. Check current directory for existing `.graphiti.json`
2. Detect project type (development/learning/research/other)
3. Generate or update configuration
4. Store configuration to memory

## Project Type Detection

| Type | Indicators | Example |
| --- | --- | --- |
| Development | Cargo.toml, package.json, go.mod, pom.xml, .rs, .py, .go, .java, .ts | Software projects |
| Learning | .md, .pdf, course/, notes/, 笔记/, 高考/, tutorial/ | Courses, tutorials |
| Research | papers/, 实验/, 数据/, dataset/, research/ | Academic, scientific |
| Personal | 笔记/, 日记/, ideas/, knowledge/ | Personal knowledge |

## Workflow

### Step 1: Check Existing Config

Check if `.graphiti.json` already exists:

- If exists: Ask user to update or skip
- If missing: Proceed to detection

### Step 2: Detect Project Type

Scan directory for indicators:

- Development: Source code files, package managers (Cargo.toml, package.json, go.mod, etc.)
- Learning: Notes, PDFs, course materials (.md, .pdf, course/, 笔记/)
- Research: Papers, experiments, datasets (papers/, 实验/, data/, dataset/)
- Personal: Personal notes, ideas (笔记/, 日记/, ideas/, knowledge/)

### Step 3: Generate Configuration

Invoke `graphiti-config-generator` agent with Task tool:

```
Task(
    description="Generate Graphiti config for [project-type]",
    prompt="Generate .graphiti.json for a [project-type] project in current directory.\n- Project name: [directory-name]\n- Project type: [development/learning/research/personal]\n\nFollow the workflow in graphiti-config-generator.agent.md",
    subagent_type="graphiti-config-generator"
)
```

### Step 4: Verify & Store

1. Validate generated JSON
2. Write to `.graphiti.json`
3. Store to memory for future reference

## Triggers

This skill activates when user mentions:

- "配置记忆" / "setup memory"
- "生成 graphiti 配置" / "generate graphiti config"
- "创建 graphiti" / "create .graphiti.json"
- Any request involving Graphiti memory setup

## Resources

See references/ for detailed configuration templates.
