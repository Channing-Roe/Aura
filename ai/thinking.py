# ============================================
# FILE: ai/thinking.py
# Deep reasoning — now uses centralized LLMClient
# ============================================

import hashlib
import logging
from collections import OrderedDict
from typing import Tuple
from config import OLLAMA_THINKING_MODEL
from ai.llm_client import think_call

logger = logging.getLogger(__name__)

_CACHE_MAX = 128

THINKING_SYSTEM = """You are an advanced reasoning system. Think deeply about problems step by step.
Break down the problem, consider multiple approaches, reason through implications,
and arrive at a well-reasoned conclusion. Always end with a clear 'Conclusion:' section."""


class _LRUCache:
    def __init__(self, maxsize: int = _CACHE_MAX):
        self._store   = OrderedDict()
        self._maxsize = maxsize
        import threading
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def set(self, key: str, value):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)


class ThinkingSystem:
    """Advanced reasoning using a dedicated reasoning model with LRU caching."""

    def __init__(self):
        self._cache = _LRUCache(maxsize=_CACHE_MAX)

    def deep_think(self, problem: str, context: str = "") -> Tuple[str, str]:
        print("🧠 Deep thinking mode activated...")

        cache_key = hashlib.md5(f"{problem}{context}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            print("💭 Using cached thoughts")
            return cached

        prompt = f"""Problem: {problem}

Context: {context}

Think through this carefully:
1. Break down the problem
2. Consider different approaches
3. Reason through implications
4. Arrive at a well-reasoned conclusion

Provide your reasoning process and final answer."""

        reasoning = think_call(OLLAMA_THINKING_MODEL, prompt, system=THINKING_SYSTEM)

        if not reasoning:
            return "", "Thinking failed — Ollama did not respond."

        if "conclusion:" in reasoning.lower():
            parts            = reasoning.lower().split("conclusion:")
            thinking_process = reasoning[:len(parts[0])].strip()
            conclusion       = reasoning[len(parts[0]) + len("conclusion:"):].strip()
        else:
            thinking_process = reasoning
            conclusion       = reasoning

        result = (thinking_process, conclusion)
        self._cache.set(cache_key, result)
        return result
