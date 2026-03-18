# =============================================================================
# FILE: ai/decision.py  (UPDATED — adds skill routing, browser, scheduler)
#
# PATCH INSTRUCTIONS:
#   Replace your existing ai/decision.py with this file.
#   Changes from original:
#     1. Added skill registry check BEFORE LLM routing (fast path)
#     2. Added "browser_use" route flag
#     3. Added "schedule" route flag
#     4. decide_and_execute() handles the new flags
# =============================================================================

import json
import re
import logging
from typing import Dict
from config import OLLAMA_MODEL
from ai.llm_client import route_call

logger = logging.getLogger(__name__)

_ROUTE_SYSTEM = """You are a tool router. Return ONLY a raw JSON object with no markdown, no explanation.
Set each tool to true only if the user explicitly needs it."""

_ROUTE_TEMPLATE = """{
  "code_generation": false,
  "web_search": false,
  "deep_research": false,
  "deep_thinking": false,
  "image_generation": false,
  "face_recognition": false,
  "vision_analysis": false,
  "music_recognition": false,
  "computer_use": false,
  "browser_use": false,
  "schedule": false
}"""

# Schedule trigger keywords
_SCHEDULE_KWS = [
    "remind me", "schedule", "every day", "every hour", "every week",
    "every monday", "every morning", "set an alarm", "alert me", "notify me",
    "in 10 minutes", "in 30 minutes", "every night", "daily at", "cron"
]

# Browser trigger keywords
_BROWSER_KWS = [
    "go to", "open website", "browse to", "visit", "fill in the form",
    "log in to", "click on", "scrape", "web automation", "fill out"
]


class DecisionSystem:
    def __init__(self, tool_executor, thinking_system,
                 coding_system=None, face_system=None, music_system=None):
        self.tools        = tool_executor
        self.thinking     = thinking_system
        self.coding       = coding_system
        self.face         = face_system
        self.music_system = music_system

    def ai_route(self, prompt: str, context: str) -> Dict[str, bool]:
        routing_prompt = (
            f"{_ROUTE_TEMPLATE}\n\n"
            f"Only set tools to true if explicitly needed.\n"
            f"User request: {prompt}"
        )
        raw = route_call(OLLAMA_MODEL, routing_prompt)
        if raw:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Route response was not valid JSON — using keyword fallback")

        # Keyword fallback
        low = prompt.lower()
        if any(kw in low for kw in ['write', 'make', 'create', 'code', 'python', 'script']):
            return {"code_generation": True}
        if any(kw in low for kw in _SCHEDULE_KWS):
            return {"schedule": True}
        if any(kw in low for kw in _BROWSER_KWS):
            return {"browser_use": True}
        from ai.computer_use import should_use_computer
        if should_use_computer(prompt):
            return {"computer_use": True}
        return {}

    def decide_and_execute(self, prompt: str, context: str) -> Dict[str, any]:
        actions = {
            'thinking_used':    False,  'web_used':          False,
            'research_used':    False,  'vision_used':        False,
            'image_generated':  False,  'code_generated':     False,
            'face_recognized':  False,  'thinking_result':    '',
            'web_result':       '',     'research_result':    '',
            'vision_result':    '',     'image_result':       '',
            'code_result':      {},     'face_result':        {},
            'computer_used':    False,  'computer_result':    '',
            # New fields
            'skill_used':       False,  'skill_result':       '',
            'browser_used':     False,  'browser_result':     '',
            'schedule_used':    False,  'schedule_result':    '',
        }

        # ── 0. Skills fast-path (before LLM routing) ──────────────────────────
        try:
            from config import load_config
            cfg = load_config()
            if cfg.get("enable_skills", True):
                from skills.skill_loader import get_registry
                skill_result = get_registry().execute(prompt, context)
                if skill_result:
                    actions['skill_used']   = True
                    actions['skill_result'] = skill_result.output
                    logger.info(f"Skill handled: {skill_result.skill}")
                    # Skills can fully handle the request — return early
                    return actions
        except Exception as e:
            logger.debug(f"Skill routing skipped: {e}")

        # ── 1. Scheduler shortcut ─────────────────────────────────────────────
        low = prompt.lower()
        if any(kw in low for kw in _SCHEDULE_KWS):
            try:
                from scheduler import get_scheduler
                sched = get_scheduler()
                if not sched._started:
                    sched.start()
                result = sched.schedule_from_text(prompt)
                actions['schedule_used']   = True
                actions['schedule_result'] = result
                return actions
            except Exception as e:
                logger.warning(f"Scheduler error: {e}")

        # ── 2. Normal LLM routing ─────────────────────────────────────────────
        from ai.computer_use import should_use_computer
        route = {"computer_use": True} if should_use_computer(prompt) \
            else self.ai_route(prompt, context)

        # ── Browser Use ───────────────────────────────────────────────────────
        if route.get("browser_use"):
            try:
                from tools.browser import browser_task
                result = browser_task(prompt, context)
                actions['browser_used']   = True
                actions['browser_result'] = result.output
            except Exception as e:
                actions['browser_result'] = f"Browser error: {e}"
            return actions

        # ── Computer Use ──────────────────────────────────────────────────────
        if route.get("computer_use"):
            try:
                from ai.computer_use import execute_computer_task
                result = execute_computer_task(prompt)
                actions['computer_used']   = True
                actions['computer_result'] = result
            except Exception as e:
                actions['computer_result'] = f"Computer use error: {e}"
            return actions

        # ── Existing tool handlers (web search, code, vision, etc.) ──────────
        if route.get("web_search") and self.tools:
            try:
                from tools.web_search import web_search
                actions['web_result'] = web_search(prompt)
                actions['web_used']   = True
            except Exception as e:
                logger.warning(f"Web search error: {e}")

        if route.get("deep_research") and self.tools:
            try:
                from tools.web_search import deep_research
                actions['research_result'] = deep_research(prompt)
                actions['research_used']   = True
            except Exception as e:
                logger.warning(f"Deep research error: {e}")

        if route.get("deep_thinking") and self.thinking:
            try:
                actions['thinking_result'] = self.thinking.think(prompt)
                actions['thinking_used']   = True
            except Exception as e:
                logger.warning(f"Thinking error: {e}")

        if route.get("code_generation") and self.coding:
            try:
                actions['code_result']    = self.coding.generate(prompt)
                actions['code_generated'] = True
            except Exception as e:
                logger.warning(f"Code gen error: {e}")

        if route.get("image_generation") and self.tools:
            try:
                from tools.image_gen import generate_image_local
                actions['image_result']    = generate_image_local(prompt)
                actions['image_generated'] = True
            except Exception as e:
                logger.warning(f"Image gen error: {e}")

        if route.get("vision_analysis"):
            try:
                from ai.vision import capture_and_describe
                actions['vision_result'] = capture_and_describe()
                actions['vision_used']   = True
            except Exception as e:
                logger.warning(f"Vision error: {e}")

        if route.get("music_recognition") and self.music_system:
            try:
                result = self.music_system.recognize()
                if result:
                    actions['web_result'] = str(result)
                    actions['web_used']   = True
            except Exception as e:
                logger.warning(f"Music recognition error: {e}")

        return actions
