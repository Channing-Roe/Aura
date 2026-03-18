# =============================================================================
# FILE: skills/skill_loader.py
# AURA Skills Plugin System
#
# Drop a folder into skills/ with a skill.py + SKILL.md and it auto-loads.
# Skills can define:
#   - Trigger keywords/phrases (for keyword routing)
#   - A run(prompt, context) function (entry point)
#   - Optional metadata: icon, description, author, version
#
# Usage:
#   registry = get_registry()
#   result   = registry.execute("weather in London", context)
# =============================================================================

import os
import sys
import json
import logging
import importlib.util
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SkillMeta:
    """Metadata parsed from a skill's SKILL.md or skill.py module attributes."""
    name:        str
    description: str        = ""
    author:      str        = "community"
    version:     str        = "1.0.0"
    icon:        str        = "🔧"
    keywords:    List[str]  = field(default_factory=list)
    enabled:     bool       = True
    path:        str        = ""


@dataclass
class SkillResult:
    success:   bool
    output:    str
    skill:     str
    data:      Any = None   # optional structured payload


# ── Skill loader ──────────────────────────────────────────────────────────────

class Skill:
    """
    Wraps a single loaded skill module.
    Each skill directory must contain skill.py with at minimum:
        KEYWORDS = ["trigger", "words"]
        def run(prompt: str, context: str = "") -> str: ...
    """

    def __init__(self, meta: SkillMeta, module):
        self.meta   = meta
        self._mod   = module

    def matches(self, prompt: str) -> bool:
        """Return True if this skill should handle the prompt."""
        low = prompt.lower()
        # Check module-level match() override first
        if hasattr(self._mod, 'match'):
            try:
                return bool(self._mod.match(prompt))
            except Exception:
                pass
        # Keyword matching
        return any(kw.lower() in low for kw in self.meta.keywords)

    def run(self, prompt: str, context: str = "") -> SkillResult:
        """Execute the skill and return a SkillResult."""
        if not hasattr(self._mod, 'run'):
            return SkillResult(success=False, output="Skill has no run() function", skill=self.meta.name)
        try:
            result = self._mod.run(prompt, context)
            if isinstance(result, dict):
                return SkillResult(
                    success=result.get('success', True),
                    output=result.get('output', str(result)),
                    skill=self.meta.name,
                    data=result.get('data')
                )
            return SkillResult(success=True, output=str(result), skill=self.meta.name)
        except Exception as e:
            logger.error(f"Skill '{self.meta.name}' raised: {e}", exc_info=True)
            return SkillResult(success=False, output=f"Skill error: {e}", skill=self.meta.name)


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Scans skills/ subdirectories and loads every valid skill.
    Thread-safe after initial load.
    """

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._dir    = skills_dir
        self._skills: Dict[str, Skill] = {}
        self._loaded = False

    # ── Discovery ────────────────────────────────────────────────────────────

    def load_all(self) -> int:
        """
        Scan the skills directory and load every skill subdirectory.
        Returns the number of successfully loaded skills.
        """
        count = 0
        if not self._dir.exists():
            logger.warning(f"Skills directory not found: {self._dir}")
            return 0

        for entry in sorted(self._dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith('_') or entry.name.startswith('.'):
                continue
            skill_py = entry / "skill.py"
            if not skill_py.exists():
                continue
            try:
                skill = self._load_skill(entry)
                if skill and skill.meta.enabled:
                    self._skills[skill.meta.name] = skill
                    logger.info(f"✅ Loaded skill: {skill.meta.icon} {skill.meta.name}")
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load skill '{entry.name}': {e}")

        self._loaded = True
        logger.info(f"Skills loaded: {count} skills active")
        return count

    def _load_skill(self, skill_dir: Path) -> Optional[Skill]:
        """Load a single skill from a directory."""
        skill_py = skill_dir / "skill.py"

        # Load the Python module
        spec = importlib.util.spec_from_file_location(
            f"skills.{skill_dir.name}", skill_py
        )
        mod = importlib.util.module_from_spec(spec)
        # Give the skill module access to its own directory
        sys.modules[spec.name] = mod
        mod.__skill_dir__ = str(skill_dir)
        spec.loader.exec_module(mod)

        # Extract metadata from module attributes or SKILL.md
        meta = self._parse_meta(skill_dir, mod)
        return Skill(meta, mod)

    def _parse_meta(self, skill_dir: Path, mod) -> SkillMeta:
        """Build SkillMeta from module attributes with SKILL.md as fallback."""
        name = getattr(mod, 'NAME', skill_dir.name.replace('_', ' ').title())

        # Parse SKILL.md for description if present
        description = getattr(mod, 'DESCRIPTION', '')
        skill_md = skill_dir / "SKILL.md"
        if not description and skill_md.exists():
            try:
                lines = skill_md.read_text(encoding='utf-8').splitlines()
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        description = line.strip()
                        break
            except Exception:
                pass

        # Check for disabled flag in config.json
        enabled = True
        cfg_file = skill_dir / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                enabled = cfg.get('enabled', True)
            except Exception:
                pass

        return SkillMeta(
            name=name,
            description=description,
            author=getattr(mod, 'AUTHOR',   'community'),
            version=getattr(mod, 'VERSION', '1.0.0'),
            icon=getattr(mod,    'ICON',    '🔧'),
            keywords=getattr(mod, 'KEYWORDS', []),
            enabled=enabled,
            path=str(skill_dir),
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def find(self, prompt: str) -> Optional[Skill]:
        """Return the first skill that matches the prompt, or None."""
        if not self._loaded:
            self.load_all()
        for skill in self._skills.values():
            if skill.matches(prompt):
                return skill
        return None

    def execute(self, prompt: str, context: str = "") -> Optional[SkillResult]:
        """
        Find and run the best matching skill.
        Returns None if no skill matches — caller falls through to LLM.
        """
        skill = self.find(prompt)
        if not skill:
            return None
        logger.info(f"Dispatching to skill: {skill.meta.name}")
        return skill.run(prompt, context)

    def execute_by_name(self, name: str, prompt: str, context: str = "") -> Optional[SkillResult]:
        """Run a specific skill by name."""
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(success=False, output=f"No skill named '{name}'", skill=name)
        return skill.run(prompt, context)

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_skills(self) -> List[SkillMeta]:
        if not self._loaded:
            self.load_all()
        return [s.meta for s in self._skills.values()]

    def reload(self):
        """Hot-reload all skills (useful for development)."""
        self._skills.clear()
        self._loaded = False
        self.load_all()

    def install(self, skill_dir: Path) -> bool:
        """
        Install a skill from an arbitrary directory by copying it into skills/.
        Returns True on success.
        """
        import shutil
        dest = self._dir / skill_dir.name
        try:
            shutil.copytree(str(skill_dir), str(dest))
            skill = self._load_skill(dest)
            if skill:
                self._skills[skill.meta.name] = skill
                logger.info(f"Installed skill: {skill.meta.name}")
                return True
        except Exception as e:
            logger.error(f"Skill install failed: {e}")
        return False

    def enable(self, name: str, enabled: bool = True):
        """Enable or disable a skill at runtime."""
        if name in self._skills:
            self._skills[name].meta.enabled = enabled
            # Persist to config.json
            cfg_file = Path(self._skills[name].meta.path) / "config.json"
            try:
                cfg = {}
                if cfg_file.exists():
                    cfg = json.loads(cfg_file.read_text())
                cfg['enabled'] = enabled
                cfg_file.write_text(json.dumps(cfg, indent=2))
            except Exception:
                pass


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[SkillRegistry] = None

def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _registry.load_all()
    return _registry
