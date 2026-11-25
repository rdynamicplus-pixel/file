---
name: agent-skills
description: Reference for creating, structuring, installing and auditing Agent Skills (SKILL.md folders) across Claude Code, claude.ai and the Claude API. Use when the user wants to create a new skill, install or register a skill, fix a skill's frontmatter, decide where a skill should live, or asks how skills load, what limits apply, or whether a skill is safe to use.
---

# Agent Skills

Skills are filesystem folders that package instructions, metadata and optional resources (scripts, templates, references). Claude loads them on demand, so guidance doesn't have to be repeated in every conversation.

## How a skill loads (progressive disclosure)

| Level | When loaded | Cost | Content |
| --- | --- | --- | --- |
| 1. Metadata | Always, at startup | ~100 tokens | `name` + `description` from frontmatter |
| 2. Instructions | When the skill triggers | keep under ~5k tokens | Body of `SKILL.md` |
| 3. Resources | Only when needed | 0 until read | Extra `.md` files read on demand; scripts are run, only their output enters context |

Consequences:
- The `description` is the only thing matched against the request, so it must say **what the skill does and when to use it**.
- Keep `SKILL.md` lean; move long references into sibling files (`REFERENCE.md`, `FORMS.md`) and link them.
- Prefer bundled scripts for deterministic steps: the code never enters context, only the output.

## Required structure

```
my-skill/
├── SKILL.md          # required
├── REFERENCE.md      # optional, read when linked
└── scripts/
    └── tool.py       # optional, run via bash
```

```markdown
---
name: my-skill
description: What it does. Use when <trigger situations / keywords>.
---

# My Skill

## Instructions
Step-by-step guidance.

## Examples
Concrete examples.
```

### Frontmatter rules
- `name`: required, max 64 chars, only lowercase letters, digits and hyphens, no XML tags, must not contain the reserved words `anthropic` or `claude`.
- `description`: required, non-empty, max 1,024 chars, no XML tags, states both what and when.

Validate these before saving a new skill.

## Where to install

| Surface | How | Scope |
| --- | --- | --- |
| Claude Code (project) | `.claude/skills/<name>/SKILL.md` in the repo | Everyone using the repo |
| Claude Code (personal) | `~/.claude/skills/<name>/SKILL.md` | Just this user |
| Claude Code (plugin) | Ship inside a plugin | Plugin users |
| claude.ai | Upload the folder as a zip in Settings (Pro/Max/Ultra/Team/Enterprise with code execution on) | The account |
| Claude API | Upload via `/v1/skills`, then list `type` + `skill_id` in `container.skills` and include the code execution tool | Whole workspace |
| Managed Agents | Agent's `skills` array, or a GitHub repo mounted on the session | The agent |

Skills **do not sync** between surfaces: install separately on each.

Pre-built document skills (`pptx`, `xlsx`, `docx`, `pdf`) exist on claude.ai and the API, not in Claude Code.

## Runtime constraints by surface
- **Claude API**: no network access, no runtime package installs; only pre-installed packages.
- **claude.ai**: network access varies with user/admin settings.
- **Claude Code**: full network access like any local program; install packages locally, not globally.

## Security checklist (before installing any skill)
Only use skills you wrote or got from a trusted source. For anything else, audit every file:
- Unexpected network calls or URLs fetched at runtime (fetched content can carry injected instructions).
- File access outside the skill's stated purpose, reads of credentials or secrets.
- Scripts whose behaviour doesn't match the description.
- Instructions that push Claude to call tools in ways unrelated to the skill's purpose.
Treat installing a skill like installing software.

## Data retention
Skills are not ZDR-eligible; skill definitions and execution data follow standard retention.

## Workflow: creating a skill for the user
1. Ask/infer the task and its trigger situations.
2. Pick a valid `name`; write a `description` with what + when.
3. Write a concise `SKILL.md` body (steps, examples); push bulky material into linked files and deterministic steps into `scripts/`.
4. Place it in the right location for the surface (table above).
5. Check the frontmatter rules and run the security checklist.

## Further reading
- Overview: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- Authoring best practices: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- Skills API guide: https://platform.claude.com/docs/en/build-with-claude/skills-guide
- Claude Code skills: https://code.claude.com/docs/en/skills
- Open-source skills: https://github.com/anthropics/skills
