# 🔧 How to Write an AURA Skill

Skills are the plugin layer for AURA. Drop a folder into `skills/` and AURA will
discover and load it automatically on next start (or after `reload skills`).

---

## Minimal skill structure

```
skills/
└── my_skill/
    ├── skill.py     ← required: entry point
    ├── SKILL.md     ← recommended: documentation
    └── config.json  ← optional: { "enabled": true }
```

---

## skill.py contract

```python
# Required metadata
NAME        = "My Skill"
DESCRIPTION = "One-line description shown in the skill panel"
AUTHOR      = "your-name"
VERSION     = "1.0.0"
ICON        = "🚀"          # single emoji shown in UI

# Keyword triggers (any of these in the prompt activates the skill)
KEYWORDS = ["trigger word", "another phrase"]


# Optional: full control over matching (overrides KEYWORDS if present)
def match(prompt: str) -> bool:
    return "my trigger" in prompt.lower()


# Required: the actual skill logic
def run(prompt: str, context: str = "") -> dict:
    # context contains AURA's current conversation context
    return {
        "success": True,
        "output":  "The text AURA will speak/display",
        "data":    {}   # optional structured data for GUI / other agents
    }
    # Alternatively, return a plain string:
    # return "my response"
```

---

## Tips

- **No API keys hardcoded** — read from `os.environ` and document in your SKILL.md
- **Always handle exceptions** — wrap network/IO in try/except and return `success: False`
- **Keep it fast** — skills run synchronously before the LLM. Aim for < 3 seconds.
- **Use `__skill_dir__`** — your module's `__skill_dir__` attribute points to your
  skill folder so you can load local data files:
  ```python
  import os, json
  data = json.load(open(os.path.join(__skill_dir__, "data.json")))
  ```

---

## Skills that ship with AURA

| Skill           | Trigger examples                             |
|-----------------|----------------------------------------------|
| 🌤️ Weather      | "weather in Paris", "temperature in Berlin"  |
| 📐 Unit Convert | "100 km in miles", "72°F to Celsius"         |

---

## Sharing your skill

1. Fork the AURA repo
2. Add your folder under `skills/`
3. Open a PR with a clear SKILL.md

Community skills that pass review get listed in the Skills Marketplace panel.
