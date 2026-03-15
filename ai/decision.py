# ============================================
# FILE: ai/decision.py
# Tool router — now uses centralized LLMClient
# ============================================

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
  "computer_use": false
}"""


class DecisionSystem:
    def __init__(self, tool_executor, thinking_system,
                 coding_system=None, face_system=None, music_system=None):
        self.tools        = tool_executor
        self.thinking     = thinking_system
        self.coding       = coding_system
        self.face         = face_system
        self.music_system = music_system

    def ai_route(self, prompt: str, context: str) -> Dict[str, bool]:
        """Use LLM to decide which tools to activate. Returns a dict of bool flags."""
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
        from ai.computer_use import should_use_computer
        if should_use_computer(prompt):
            return {"computer_use": True}
        return {}

    def decide_and_execute(self, prompt: str, context: str) -> Dict[str, any]:
        actions = {
            'thinking_used': False,  'web_used': False,       'research_used': False,
            'vision_used':   False,  'image_generated': False, 'code_generated': False,
            'face_recognized': False, 'thinking_result': '',   'web_result': '',
            'research_result': '',   'vision_result': '',      'image_result': '',
            'code_result': {},       'face_result': {},
            'computer_used': False,  'computer_result': '',
        }

        from ai.computer_use import should_use_computer
        route = {"computer_use": True} if should_use_computer(prompt) \
            else self.ai_route(prompt, context)

        # ── Computer Use ───────────────────────────────────────────────────
        if route.get("computer_use"):
            print("🖥️  Activating computer use...")
            try:
                from ai.computer_use import get_computer_agent
                result = get_computer_agent().run(prompt)
                actions['computer_used']   = True
                actions['computer_result'] = result
            except Exception as e:
                logger.error(f"Computer use failed: {e}")
                actions['computer_used']   = True
                actions['computer_result'] = f"Computer use failed: {e}"
            return actions  # short-circuit

        # ── Code Generation ────────────────────────────────────────────────
        if route.get("code_generation") and self.coding:
            print("💻 Generating code...")
            actions['code_generated'] = True
            actions['code_result']    = self.coding.generate_and_save(prompt, context)

        # ── Web Search ─────────────────────────────────────────────────────
        if route.get("web_search"):
            print("🌐 Web search...")
            actions['web_used']   = True
            actions['web_result'] = self.tools.web_search(prompt)

        # ── Deep Research ──────────────────────────────────────────────────
        if route.get("deep_research"):
            print("📚 Deep research...")
            actions['research_used']   = True
            actions['research_result'] = self.tools.deep_research(prompt)

        # ── Deep Thinking ──────────────────────────────────────────────────
        if route.get("deep_thinking"):
            print("🧠 Deep thinking...")
            _, conclusion              = self.thinking.deep_think(prompt, context)
            actions['thinking_used']   = True
            actions['thinking_result'] = conclusion

        # ── Image Generation ───────────────────────────────────────────────
        if route.get("image_generation"):
            print("🎨 Generating image...")
            actions['image_generated'] = True
            actions['image_result']    = self.tools.generate_image_local(prompt)[0]

        # ── Vision Analysis ────────────────────────────────────────────────
        if route.get("vision_analysis"):
            print("📷 Vision analysis...")
            from ai.vision import get_visual_context
            actions['vision_used']   = True
            actions['vision_result'] = get_visual_context(force=True)

        # ── Music Recognition ──────────────────────────────────────────────
        if route.get("music_recognition") and self.music_system:
            print("🎵 Music recognition...")
            actions['music_used']   = True
            actions['music_result'] = self.music_system.recognize()

        return actions
