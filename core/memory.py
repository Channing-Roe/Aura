"""
core/memory.py — AURA Memory System v3

Two completely separate stores:

  LONG-TERM KNOWLEDGE (aura_knowledge.json + aura_knowledge_embeddings.npy)
  ──────────────────────────────────────────────────────────────────────────
  Facts, preferences, and things AURA learns about you permanently.
  Never expires. Survives restarts. Only injected when relevant to what
  you are asking RIGHT NOW — not blindly dumped into every prompt.

  SHORT-TERM SESSION (in-memory only, resets on restart)
  ──────────────────────────────────────────────────────
  The last N turns of the CURRENT conversation.
  Completely isolated per session — old chats never bleed through.

  CONVERSATION LOG (aura_memory.json — archive only)
  ───────────────────────────────────────────────────
  Full history written to disk but NEVER auto-injected.
  Only used when you explicitly ask AURA to recall something old.

Result:
  - AURA always knows your name, preferences, projects → injected when relevant
  - AURA never randomly brings up a 3-week-old chat
  - Each new session starts clean
  - Fast: embeddings live in a .npy matrix, not buried inside a 171KB JSON
"""

import json
import os
import re
import logging
import threading
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────────────────────
MEMORY_FILE        = "aura_memory.json"           # conversation archive (never auto-injected)
KNOWLEDGE_FILE     = "aura_knowledge.json"         # long-term facts
KNOWLEDGE_EMB_FILE = "aura_knowledge_embeddings.npy"  # embeddings stored separately

# ── Limits ─────────────────────────────────────────────────────────────────────
MAX_SESSION_TURNS   = 12    # turns kept in current-session context
MAX_LOG_ENTRIES     = 2000  # max entries in conversation archive before pruning
MAX_KNOWLEDGE_FACTS = 500   # max facts in long-term store
MAX_CONTEXT_FACTS   = 8     # max facts injected per LLM call
FACT_SIM_THRESHOLD  = 0.55  # minimum similarity score to inject a fact

# ── Thread locks ───────────────────────────────────────────────────────────────
_save_lock      = threading.Lock()
_knowledge_lock = threading.Lock()

# ── In-memory session (never written to disk) ──────────────────────────────────
_session_turns: List[Dict] = []

# ── Knowledge cache ────────────────────────────────────────────────────────────
_knowledge_facts:      List[Dict]           = []
_knowledge_embeddings: Optional[np.ndarray] = None  # shape (N, 384)


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

_embedding_model      = None
_embedding_model_lock = threading.Lock()


def _get_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed — "
                           "knowledge search disabled. pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Embedding model load error: {e}")
    return _embedding_model


def _embed(text: str) -> Optional[np.ndarray]:
    model = _get_model()
    if not model or not text.strip():
        return None
    try:
        return model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Embed error: {e}")
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM KNOWLEDGE STORE
# ══════════════════════════════════════════════════════════════════════════════

def _load_knowledge():
    global _knowledge_facts, _knowledge_embeddings
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                _knowledge_facts = json.load(f)
            logger.info(f"Loaded {len(_knowledge_facts)} knowledge facts")
        except Exception as e:
            logger.error(f"Knowledge load error: {e}")
            _knowledge_facts = []
    else:
        _knowledge_facts = []

    if os.path.exists(KNOWLEDGE_EMB_FILE) and _knowledge_facts:
        try:
            _knowledge_embeddings = np.load(KNOWLEDGE_EMB_FILE)
            if _knowledge_embeddings.shape[0] != len(_knowledge_facts):
                logger.warning("Embedding count mismatch — rebuilding")
                _rebuild_embeddings()
        except Exception as e:
            logger.error(f"Embeddings load error: {e}")
            _rebuild_embeddings()


def _save_knowledge():
    with _knowledge_lock:
        tmp = KNOWLEDGE_FILE + ".tmp"
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(_knowledge_facts, f, indent=2, ensure_ascii=False)
            os.replace(tmp, KNOWLEDGE_FILE)
        except Exception as e:
            logger.error(f"Knowledge save error: {e}")
            return
        if _knowledge_embeddings is not None and len(_knowledge_embeddings) > 0:
            try:
                np.save(KNOWLEDGE_EMB_FILE, _knowledge_embeddings)
            except Exception as e:
                logger.error(f"Embeddings save error: {e}")


def _rebuild_embeddings():
    global _knowledge_embeddings
    if not _knowledge_facts:
        _knowledge_embeddings = None
        return
    model = _get_model()
    if not model:
        _knowledge_embeddings = None
        return
    try:
        texts = [f['text'] for f in _knowledge_facts]
        _knowledge_embeddings = model.encode(texts, convert_to_numpy=True,
                                              show_progress_bar=False, batch_size=32)
        logger.info(f"Rebuilt embeddings for {len(texts)} facts")
    except Exception as e:
        logger.error(f"Embedding rebuild error: {e}")
        _knowledge_embeddings = None


# Category detection
_CATEGORY_HINTS = {
    'name':        ['my name is', 'call me', "i'm called", 'i go by'],
    'preference':  ['i prefer', 'i like', 'i love', "i don't like", 'i hate',
                    'i always', 'i usually', 'my favourite', 'i enjoy'],
    'skill':       ['i know', 'i use', 'i work with', 'i code in',
                    'i develop', 'i build', 'i program'],
    'project':     ['my project', 'i am building', "i'm working on",
                    'my app', 'my software', 'my tool', 'my bot'],
    'personal':    ['i am', "i'm", 'i live', 'my job', 'i work as',
                    'i study', 'my os', 'my computer', 'my pc'],
    'instruction': ['always', 'never', 'from now on', 'stop doing',
                    'please always', "don't ever", 'remember to'],
}


def _categorise(text: str) -> str:
    low = text.lower()
    for cat, hints in _CATEGORY_HINTS.items():
        if any(h in low for h in hints):
            return cat
    return 'general'


def learn(fact: str, category: str = None, source: str = 'user') -> bool:
    """
    Store a fact in long-term knowledge permanently.
    Deduplicates — if >80% similar to existing fact, reinforces instead.
    Returns True if added as new, False if reinforced existing.
    """
    global _knowledge_embeddings
    if not fact.strip():
        return False

    new_emb = _embed(fact)

    # Deduplication check
    if new_emb is not None and _knowledge_embeddings is not None and len(_knowledge_embeddings) > 0:
        sims = np.dot(_knowledge_embeddings, new_emb) / (
            np.linalg.norm(_knowledge_embeddings, axis=1) * np.linalg.norm(new_emb) + 1e-9
        )
        if float(sims.max()) > 0.80:
            idx = int(sims.argmax())
            _knowledge_facts[idx]['last_reinforced'] = datetime.now().isoformat()
            _knowledge_facts[idx]['reinforcement_count'] = \
                _knowledge_facts[idx].get('reinforcement_count', 0) + 1
            _save_knowledge()
            return False

    cat = category or _categorise(fact)
    entry = {
        'text':                fact.strip(),
        'category':            cat,
        'source':              source,
        'timestamp':           datetime.now().isoformat(),
        'last_reinforced':     datetime.now().isoformat(),
        'access_count':        0,
        'reinforcement_count': 0,
    }
    _knowledge_facts.append(entry)

    if new_emb is not None:
        _knowledge_embeddings = (
            new_emb.reshape(1, -1) if _knowledge_embeddings is None or len(_knowledge_embeddings) == 0
            else np.vstack([_knowledge_embeddings, new_emb])
        )

    if len(_knowledge_facts) > MAX_KNOWLEDGE_FACTS:
        _prune_knowledge()

    _save_knowledge()
    logger.info(f"Learned [{cat}]: {fact[:80]}")
    return True


def forget(text: str) -> bool:
    """Remove facts matching text from long-term knowledge."""
    global _knowledge_facts, _knowledge_embeddings
    low = text.lower()
    keep = [i for i, f in enumerate(_knowledge_facts) if low not in f['text'].lower()]
    if len(keep) == len(_knowledge_facts):
        return False
    _knowledge_facts = [_knowledge_facts[i] for i in keep]
    if _knowledge_embeddings is not None and len(_knowledge_embeddings) > 0:
        arr = _knowledge_embeddings[keep]
        _knowledge_embeddings = arr if len(arr) > 0 else None
    _save_knowledge()
    logger.info(f"Forgot facts matching: {text[:60]}")
    return True


def _prune_knowledge():
    global _knowledge_facts, _knowledge_embeddings
    protected = {'name', 'instruction', 'personal'}
    prunable  = [(i, f) for i, f in enumerate(_knowledge_facts)
                 if f.get('category') not in protected]
    prunable.sort(key=lambda x: x[1].get('access_count', 0))
    excess     = len(_knowledge_facts) - MAX_KNOWLEDGE_FACTS
    remove_set = {i for i, _ in prunable[:excess]}
    keep       = [i for i in range(len(_knowledge_facts)) if i not in remove_set]
    _knowledge_facts = [_knowledge_facts[i] for i in keep]
    if _knowledge_embeddings is not None and len(_knowledge_embeddings) > 0:
        _knowledge_embeddings = _knowledge_embeddings[keep]


def get_relevant_knowledge(query: str, top_k: int = MAX_CONTEXT_FACTS) -> List[Dict]:
    """
    Return the most relevant facts for the current query.
    O(N) dot product over pre-loaded matrix — very fast even at 500 facts.
    """
    if not _knowledge_facts:
        return []

    query_emb = _embed(query)

    if query_emb is not None and _knowledge_embeddings is not None and len(_knowledge_embeddings) > 0:
        sims = np.dot(_knowledge_embeddings, query_emb) / (
            np.linalg.norm(_knowledge_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_idx = np.argsort(sims)[::-1]
        results = []
        for idx in top_idx:
            if float(sims[idx]) < FACT_SIM_THRESHOLD:
                break
            fact = dict(_knowledge_facts[idx])
            fact['_score'] = float(sims[idx])
            results.append((float(sims[idx]), int(idx), fact))
            if len(results) >= top_k:
                break
        for _, idx, _ in results:
            _knowledge_facts[idx]['access_count'] = \
                _knowledge_facts[idx].get('access_count', 0) + 1
        return [f for _, _, f in results]

    # Keyword fallback
    qwords = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    scored = []
    for i, fact in enumerate(_knowledge_facts):
        fwords = set(re.findall(r'\b[a-z]{3,}\b', fact['text'].lower()))
        score  = len(qwords & fwords) / max(len(qwords), 1)
        if score > 0.1:
            scored.append((score, fact))
    scored.sort(reverse=True)
    return [f for _, f in scored[:top_k]]


def list_all_knowledge() -> List[Dict]:
    """Return all stored facts. For GUI knowledge browser."""
    return list(_knowledge_facts)


# ══════════════════════════════════════════════════════════════════════════════
# AUTO FACT EXTRACTION
# Scans each user message for things worth learning permanently
# ══════════════════════════════════════════════════════════════════════════════

_LEARN_PATTERNS = [
    (r'\bremember\s+that\b',                           'instruction'),
    (r'\balways\s+(?:call|refer\s+to)\s+me\b',        'name'),
    (r'\bmy\s+name\s+is\b',                            'name'),
    (r'\bcall\s+me\b',                                 'name'),
    (r'\bi\s+(?:always|never|prefer|hate|love)\b',     'preference'),
    (r'\bmy\s+favourite\b',                            'preference'),
    (r"\bi(?:'m| am)\s+(?:a |an )?(?:developer|programmer|engineer|student|designer)\b", 'personal'),
    (r'\bi\s+(?:use|work\s+with|code\s+in|build\s+with)\b', 'skill'),
    (r'\bmy\s+(?:os|computer|laptop|pc|setup)\b',      'personal'),
    (r'\bmy\s+(?:project|app|software|bot|tool|game)\b', 'project'),
    (r"\bi(?:'m| am)\s+(?:building|working\s+on|creating|developing)\b", 'project'),
    (r'\bfrom\s+now\s+on\b',                           'instruction'),
    (r'\bstop\s+(?:doing|saying|being)\b',             'instruction'),
    (r"\bdon't\s+(?:ever|always)\b",                   'instruction'),
]


def extract_and_learn(user_message: str) -> List[str]:
    """Scan a user message for learnable facts. Returns list of new facts stored."""
    learned = []
    low = user_message.lower().strip()
    for pattern, category in _LEARN_PATTERNS:
        if re.search(pattern, low):
            fact = re.sub(
                r'^(hey aura[,\s]+|aura[,\s]+|ok[,\s]+|so[,\s]+)',
                '', user_message.strip(), flags=re.IGNORECASE
            ).strip()
            if len(fact) > 10:
                if learn(fact, category=category, source='auto_extract'):
                    learned.append(fact)
            break
    return learned


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY  (current conversation only — resets on restart)
# ══════════════════════════════════════════════════════════════════════════════

def add_session_turn(user: str, aura: str, meta: Dict = None):
    """Add a turn to the current session. Auto-extracts learnable facts."""
    if not user.strip() or not aura.strip():
        return
    turn = {
        'user':      user.strip(),
        'aura':      aura.strip(),
        'timestamp': datetime.now().isoformat(),
        'tools':     [],
    }
    if meta:
        turn['tools'] = [
            k.replace('_used', '').replace('_generated', '')
            for k in ('web_used', 'research_used', 'thinking_used',
                      'code_generated', 'vision_used', 'image_generated')
            if meta.get(k)
        ]
    _session_turns.append(turn)
    if len(_session_turns) > MAX_SESSION_TURNS:
        del _session_turns[:-MAX_SESSION_TURNS]
    extract_and_learn(user)


def clear_session():
    """Clear session memory. Call on startup so old sessions never bleed through."""
    _session_turns.clear()
    logger.info("Session memory cleared")


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION LOG  (archive — never auto-injected)
# ══════════════════════════════════════════════════════════════════════════════

def log_conversation(user: str, aura: str, meta: Dict = None):
    """Append a turn to the permanent conversation archive."""
    entry = {
        'user':      user.strip(),
        'aura':      aura.strip(),
        'timestamp': datetime.now().isoformat(),
        'session':   datetime.now().strftime("%Y%m%d_%H"),
    }
    if meta:
        entry['tools'] = [
            k.replace('_used', '').replace('_generated', '')
            for k in ('web_used', 'research_used', 'thinking_used',
                      'code_generated', 'vision_used', 'image_generated')
            if meta.get(k)
        ]
    with _save_lock:
        log = _load_log()
        log.append(entry)
        if len(log) > MAX_LOG_ENTRIES:
            log = log[-MAX_LOG_ENTRIES:]
        _save_log(log)


def search_log(query: str, top_k: int = 5) -> List[Dict]:
    """Search conversation history. Called only on explicit recall requests."""
    log = _load_log()
    if not log:
        return []
    query_emb = _embed(query)
    qwords    = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    scored    = []
    for entry in log:
        combined = f"{entry.get('user', '')} {entry.get('aura', '')}"
        if query_emb is not None:
            emb   = _embed(combined)
            score = _cosine(query_emb, emb) if emb is not None else 0.0
        else:
            ewords = set(re.findall(r'\b[a-z]{3,}\b', combined.lower()))
            score  = len(qwords & ewords) / max(len(qwords), 1)
        if score > 0.30:
            scored.append((score, entry))
    scored.sort(reverse=True)
    return [e for _, e in scored[:top_k]]


def _load_log() -> List[Dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        logger.error(f"Conversation log corrupted: {e}")
        try:
            import shutil; shutil.copy(MEMORY_FILE, MEMORY_FILE + ".bak")
        except Exception:
            pass
        return []
    except OSError as e:
        logger.error(f"Conversation log read error: {e}")
        return []


def _save_log(log: List[Dict]):
    tmp = MEMORY_FILE + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        os.replace(tmp, MEMORY_FILE)
    except OSError as e:
        logger.error(f"Conversation log save error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTEXT BUILDER  — called by llm.py on every prompt
# ══════════════════════════════════════════════════════════════════════════════

def get_context(query: str) -> str:
    """
    Build the context string injected into every AURA prompt.

    Contains ONLY:
      1. Long-term facts relevant to THIS query (semantically matched)
      2. Current session turns (this conversation only)

    Does NOT contain:
      - Old conversations from previous sessions
      - Irrelevant facts
    """
    parts = []

    relevant = get_relevant_knowledge(query)
    if relevant:
        parts.append("--- WHAT I KNOW ABOUT YOU ---")
        for fact in relevant:
            parts.append(f"  [{fact.get('category', 'general')}] {fact['text']}")
        parts.append("")

    if _session_turns:
        parts.append("--- THIS CONVERSATION ---")
        for turn in _session_turns:
            tools = f" [used: {', '.join(turn['tools'])}]" if turn.get('tools') else ""
            parts.append(f"  User: {turn['user']}")
            parts.append(f"  AURA: {turn['aura'][:500]}{tools}")
            parts.append("")

    if not parts:
        return ""
    parts.append("---")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY SHIMS  (keeps existing callers working without changes)
# ══════════════════════════════════════════════════════════════════════════════

def load_memory() -> List[Dict]:
    return _load_log()

def save_memory(hist: List[Dict]):
    pass  # managed internally now

def add_memory(hist, usr: str, aura: str, typ: str = "conversation", meta: Dict = None):
    add_session_turn(usr, aura, meta)
    log_conversation(usr, aura, meta)


# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════

def memory_stats() -> Dict:
    by_cat = Counter(f.get('category', 'general') for f in _knowledge_facts)
    return {
        'knowledge_facts':   len(_knowledge_facts),
        'facts_by_category': dict(by_cat),
        'has_embeddings':    _knowledge_embeddings is not None,
        'session_turns':     len(_session_turns),
        'log_entries':       len(_load_log()),
        'top_accessed':      sorted(
            _knowledge_facts,
            key=lambda x: x.get('access_count', 0), reverse=True
        )[:5],
    }


# ── Init ───────────────────────────────────────────────────────────────────────
_load_knowledge()
clear_session()
