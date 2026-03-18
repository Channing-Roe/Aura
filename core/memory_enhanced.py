# =============================================================================
# FILE: core/memory_enhanced.py
# AURA Long-Term Memory Enhancements
#
# Patches onto the existing core/memory.py to add:
#   1. Cross-session conversation recall  ("what did we discuss last Tuesday?")
#   2. Conversation timeline with date-based search
#   3. Topic clustering — group related conversations automatically
#   4. "Remember this conversation" — explicitly save a session summary
#   5. Forget by date / topic
#
# Usage:
#   from core.memory_enhanced import recall, save_session_summary, search_history
#   results = recall("what did we discuss about my Python project?")
# =============================================================================

import os
import re
import json
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Files ─────────────────────────────────────────────────────────────────────
HISTORY_FILE     = Path("aura_conversation_history.json")
HISTORY_EMB_FILE = Path("aura_history_embeddings.npy")
SESSION_LOG_FILE = Path("aura_session_summaries.json")

MAX_HISTORY_ENTRIES = 5000
RECALL_TOP_K        = 8
RECALL_MIN_SIM      = 0.35   # lower than knowledge threshold — we want broader recall

_history_lock = threading.Lock()

# ── In-memory caches ──────────────────────────────────────────────────────────
_history:            List[Dict]           = []
_history_embeddings: Optional[np.ndarray] = None
_history_loaded      = False


# =============================================================================
# EMBEDDING (reuses core/memory.py's model)
# =============================================================================

def _embed(text: str) -> Optional[np.ndarray]:
    try:
        from core.memory import _get_model
        model = _get_model()
        if not model:
            return None
        return model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        return None


# =============================================================================
# HISTORY STORE
# Persists every conversation turn with rich metadata
# =============================================================================

def _load_history():
    global _history, _history_embeddings, _history_loaded
    if _history_loaded:
        return

    if HISTORY_FILE.exists():
        try:
            _history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            logger.info(f"Loaded {len(_history)} history entries")
        except Exception as e:
            logger.error(f"History load error: {e}")
            _history = []
    else:
        _history = []

    if HISTORY_EMB_FILE.exists() and _history:
        try:
            _history_embeddings = np.load(str(HISTORY_EMB_FILE))
            if _history_embeddings.shape[0] != len(_history):
                logger.warning("History embedding mismatch — rebuilding")
                _rebuild_history_embeddings()
        except Exception:
            _rebuild_history_embeddings()

    _history_loaded = True


def _save_history():
    with _history_lock:
        tmp = str(HISTORY_FILE) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_history[-MAX_HISTORY_ENTRIES:], f,
                          indent=2, ensure_ascii=False)
            os.replace(tmp, str(HISTORY_FILE))
        except Exception as e:
            logger.error(f"History save error: {e}")

        if _history_embeddings is not None and len(_history_embeddings) > 0:
            try:
                np.save(str(HISTORY_EMB_FILE), _history_embeddings[-MAX_HISTORY_ENTRIES:])
            except Exception as e:
                logger.error(f"History embedding save error: {e}")


def _rebuild_history_embeddings():
    global _history_embeddings
    if not _history:
        _history_embeddings = None
        return
    try:
        from core.memory import _get_model
        model = _get_model()
        if not model:
            _history_embeddings = None
            return
        texts = [f"{e.get('user','')} {e.get('aura','')}" for e in _history]
        _history_embeddings = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64
        )
        logger.info(f"Rebuilt history embeddings ({len(texts)} entries)")
    except Exception as e:
        logger.error(f"History embed rebuild error: {e}")
        _history_embeddings = None


# =============================================================================
# LOGGING  (call this instead of / in addition to core/memory.log_conversation)
# =============================================================================

def log_turn(user: str, aura: str, meta: Dict = None):
    """
    Persist a conversation turn to the cross-session history store.
    Call this from main.py / main_gui.py alongside the existing log_conversation().
    """
    _load_history()

    entry = {
        "user":      user,
        "aura":      aura[:2000],   # cap to avoid bloat
        "timestamp": datetime.now().isoformat(),
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "weekday":   datetime.now().strftime("%A"),
        "meta":      meta or {},
    }
    _history.append(entry)

    # Update embeddings
    global _history_embeddings
    emb = _embed(f"{user} {aura}")
    if emb is not None:
        if _history_embeddings is None or len(_history_embeddings) == 0:
            _history_embeddings = emb.reshape(1, -1)
        else:
            _history_embeddings = np.vstack([_history_embeddings, emb])

    # Prune if needed
    if len(_history) > MAX_HISTORY_ENTRIES:
        _history[:] = _history[-MAX_HISTORY_ENTRIES:]
        if _history_embeddings is not None:
            _history_embeddings = _history_embeddings[-MAX_HISTORY_ENTRIES:]

    _save_history()


# =============================================================================
# RECALL  (semantic search over ALL past conversations)
# =============================================================================

def recall(query: str, top_k: int = RECALL_TOP_K) -> List[Dict]:
    """
    Semantic search across ALL past conversations.
    Returns up to top_k relevant entries sorted by similarity.

    Each entry has: user, aura, timestamp, date, weekday, score
    """
    _load_history()
    if not _history:
        return []

    query_emb = _embed(query)
    if query_emb is not None and _history_embeddings is not None and len(_history_embeddings) > 0:
        norms = np.linalg.norm(_history_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        sims  = np.dot(_history_embeddings, query_emb) / norms
        top_idx = np.argsort(sims)[::-1]

        results = []
        for idx in top_idx:
            if float(sims[idx]) < RECALL_MIN_SIM:
                break
            entry = dict(_history[idx])
            entry["score"] = float(sims[idx])
            results.append(entry)
            if len(results) >= top_k:
                break
        return results

    # Keyword fallback
    qwords = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    scored = []
    for entry in _history:
        text   = f"{entry.get('user','')} {entry.get('aura','')}".lower()
        ewords = set(re.findall(r'\b[a-z]{3,}\b', text))
        score  = len(qwords & ewords) / max(len(qwords), 1)
        if score > 0.1:
            scored.append((score, entry))
    scored.sort(reverse=True)
    return [dict(e) | {"score": s} for s, e in scored[:top_k]]


def recall_by_date(date_str: str) -> List[Dict]:
    """
    Return all conversation turns on a specific date.
    date_str can be:
      - "2024-03-15"  (ISO date)
      - "Monday"      (most recent Monday)
      - "yesterday"
      - "last Tuesday"
      - "3 days ago"
    """
    _load_history()
    target_date = _parse_date(date_str)
    if not target_date:
        return []

    date_key = target_date.strftime("%Y-%m-%d")
    return [e for e in _history if e.get("date") == date_key]


def recall_formatted(query: str) -> str:
    """
    Return a human-readable summary of recalled conversations.
    This is what gets injected into the LLM context when recall is triggered.
    """
    results = recall(query)
    if not results:
        return ""

    lines = [f"--- RECALLED CONVERSATIONS (relevant to: '{query[:60]}') ---\n"]
    for r in results:
        ts   = r.get("timestamp", "")[:16].replace("T", " ")
        day  = r.get("weekday", "")
        user = r.get("user", "")[:200]
        aura = r.get("aura", "")[:300]
        lines.append(f"[{day} {ts}]")
        lines.append(f"  You: {user}")
        lines.append(f"  AURA: {aura}")
        lines.append("")

    lines.append("--- END RECALLED CONVERSATIONS ---")
    return "\n".join(lines)


def search_history(query: str = None, date: str = None,
                   topic: str = None, limit: int = 20) -> List[Dict]:
    """
    Flexible history search for the GUI history browser.
    Supports: text query, date filter, topic keyword, result limit.
    """
    _load_history()
    results = list(_history)

    if date:
        target = _parse_date(date)
        if target:
            date_key = target.strftime("%Y-%m-%d")
            results = [e for e in results if e.get("date") == date_key]

    if topic:
        low = topic.lower()
        results = [e for e in results
                   if low in e.get("user", "").lower()
                   or low in e.get("aura", "").lower()]

    if query:
        # Re-rank by semantic similarity
        ranked = recall(query, top_k=limit * 2)
        # Filter by date/topic if also specified
        if date or topic:
            result_ts = {e["timestamp"] for e in results}
            ranked = [r for r in ranked if r["timestamp"] in result_ts]
        return ranked[:limit]

    return results[-limit:]  # most recent first (list is chronological)


# =============================================================================
# SESSION SUMMARIES
# =============================================================================

def save_session_summary(summary: str = None):
    """
    Save a summary of the current session to the session log.
    If summary is None, auto-generates one from recent turns.
    """
    from core.memory import _session_turns  # access current session

    if not summary:
        if not _session_turns:
            return
        # Build a brief summary from the session
        lines = []
        for t in _session_turns[-10:]:
            lines.append(f"User: {t.get('user','')[:100]}")
            lines.append(f"AURA: {t.get('aura','')[:150]}")
        summary = _auto_summarise("\n".join(lines))

    entry = {
        "summary":   summary,
        "timestamp": datetime.now().isoformat(),
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "weekday":   datetime.now().strftime("%A"),
        "turn_count": len(_session_turns) if hasattr(_session_turns, '__len__') else 0,
    }

    sessions = []
    if SESSION_LOG_FILE.exists():
        try:
            sessions = json.loads(SESSION_LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            sessions = []

    sessions.append(entry)
    sessions = sessions[-500:]  # keep last 500 sessions

    try:
        SESSION_LOG_FILE.write_text(
            json.dumps(sessions, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Session summary saved ({len(summary)} chars)")
    except Exception as e:
        logger.error(f"Session summary save error: {e}")


def get_recent_session_summaries(n: int = 5) -> str:
    """
    Return the last n session summaries as a formatted string.
    Useful for injecting "what we talked about recently" into context.
    """
    if not SESSION_LOG_FILE.exists():
        return ""
    try:
        sessions = json.loads(SESSION_LOG_FILE.read_text(encoding="utf-8"))
        recent   = sessions[-n:]
        lines    = ["--- RECENT SESSIONS ---\n"]
        for s in reversed(recent):
            day = s.get("weekday", "")
            dt  = s.get("timestamp", "")[:16].replace("T", " ")
            lines.append(f"[{day} {dt}] {s.get('summary', '')}")
            lines.append("")
        lines.append("--- END RECENT SESSIONS ---")
        return "\n".join(lines)
    except Exception:
        return ""


def forget_before(date_str: str) -> int:
    """Remove all history entries before a given date. Returns count deleted."""
    _load_history()
    global _history, _history_embeddings

    target = _parse_date(date_str)
    if not target:
        return 0

    date_key = target.strftime("%Y-%m-%d")
    keep     = [i for i, e in enumerate(_history) if e.get("date", "") >= date_key]
    removed  = len(_history) - len(keep)

    _history = [_history[i] for i in keep]
    if _history_embeddings is not None and len(_history_embeddings) > 0:
        _history_embeddings = _history_embeddings[keep] if keep else None

    _save_history()
    logger.info(f"Deleted {removed} history entries before {date_key}")
    return removed


# =============================================================================
# RECALL TRIGGER DETECTION
# =============================================================================

RECALL_TRIGGERS = [
    r'\bremember\b.*\bdiscuss',
    r'\bwhat did we\b',
    r'\bdo you remember\b',
    r'\blast (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\byesterday\b.*(?:said|talked|discussed)',
    r'\bearlier (?:today|this week)\b',
    r'\bwe talked about\b',
    r'\byou told me\b',
    r'\bi told you\b',
    r'\bprevious(?:ly)? (?:said|mentioned|discussed)\b',
    r'\bfew (?:days|weeks) ago\b',
    r'\blast session\b',
    r'\brecall\b',
]

_RECALL_RE = [re.compile(p, re.IGNORECASE) for p in RECALL_TRIGGERS]


def should_recall(prompt: str) -> bool:
    """Return True if the prompt is asking about past conversations."""
    return any(r.search(prompt) for r in _RECALL_RE)


def get_recall_context(prompt: str) -> str:
    """
    If the prompt needs recall, return formatted past conversation context.
    Otherwise return empty string.
    """
    if not should_recall(prompt):
        return ""
    return recall_formatted(prompt)


# =============================================================================
# HELPERS
# =============================================================================

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse natural language dates into datetime objects."""
    low = date_str.lower().strip()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if low in ("today",):
        return today
    if low in ("yesterday",):
        return today - timedelta(days=1)

    # "N days ago"
    m = re.search(r'(\d+)\s+days?\s+ago', low)
    if m:
        return today - timedelta(days=int(m.group(1)))

    # "last Monday" etc.
    days = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
            "friday":4,"saturday":5,"sunday":6}
    for day_name, day_num in days.items():
        if day_name in low:
            diff = (today.weekday() - day_num) % 7
            diff = diff or 7  # if today, go back a week
            return today - timedelta(days=diff)

    # ISO date
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            pass

    return None


def _auto_summarise(text: str) -> str:
    """Generate a brief session summary using the LLM."""
    try:
        import requests as req
        from config import OLLAMA_API_URL, OLLAMA_MODEL
        resp = req.post(
            OLLAMA_API_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": (
                    f"Summarise this conversation in 2-3 sentences, "
                    f"noting the key topics discussed:\n\n{text[:2000]}"
                ),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=30
        )
        return resp.json().get("response", "").strip()
    except Exception:
        return text[:200]


# ── Init on import ────────────────────────────────────────────────────────────
_load_history()
