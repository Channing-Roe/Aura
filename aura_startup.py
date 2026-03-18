# =============================================================================
# FILE: aura_startup.py  (v3 — updated)
# Add to top of main_gui.py:
#   from aura_startup import bootstrap_aura
#   bootstrap_aura(aura_respond_fn)
# =============================================================================

import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def bootstrap_aura(aura_respond_fn=None):
    from config import load_config
    cfg = load_config()

    _check_platform()
    if cfg.get("enable_skills",    True):  _load_skills()
    if cfg.get("enable_scheduler", True):  _start_scheduler(aura_respond_fn)
    if cfg.get("enable_telegram_bot", False) and aura_respond_fn:
        _start_telegram(aura_respond_fn)
    if cfg.get("enable_browser",   True):  _register_browser()
    if cfg.get("enable_web_dashboard", False) and aura_respond_fn:
        _start_dashboard(aura_respond_fn, cfg.get("web_dashboard_port", 5000))

    # Apply GPU config to image gen
    _apply_gpu_config(cfg)

    # Patch memory to also log to cross-session history
    _patch_memory_logging()

    logger.info("✅ AURA v3 bootstrap complete")


def _check_platform():
    try:
        from platform_compat import check_platform_deps, SYSTEM
        deps   = check_platform_deps()
        failed = [k for k, v in deps.items() if not v]
        logger.info(f"Platform: {SYSTEM} — {len(deps)-len(failed)}/{len(deps)} deps OK")
        if failed:
            logger.warning(f"Missing: {', '.join(failed)}")
    except Exception as e:
        logger.debug(f"Platform check skipped: {e}")


def _load_skills():
    try:
        from skills.skill_loader import get_registry
        n = len(get_registry().list_skills())
        logger.info(f"🔧 Skills: {n} active")
    except Exception as e:
        logger.warning(f"Skills load failed: {e}")


def _start_scheduler(fn=None):
    try:
        from scheduler import get_scheduler

        def _on_fire(job):
            logger.info(f"⏰ Scheduler: {job.name}")
            if fn:
                try:
                    fn(job.task, context="scheduled_task")
                except Exception as e:
                    logger.error(f"Scheduler job error: {e}")

        get_scheduler(on_trigger=_on_fire).start()
        logger.info("📅 Scheduler started")
    except Exception as e:
        logger.warning(f"Scheduler start failed: {e}")


def _start_telegram(fn):
    if not os.environ.get("TELEGRAM_TOKEN"):
        return
    try:
        from telegram_bot import AuraTelegramBot
        AuraTelegramBot(fn).start()
        logger.info("🤖 Telegram bot started")
    except Exception as e:
        logger.warning(f"Telegram start failed: {e}")


def _register_browser():
    try:
        from tools.browser import PLAYWRIGHT_AVAILABLE
        logger.info(f"🌐 Browser: {'ready' if PLAYWRIGHT_AVAILABLE else 'install playwright'}")
    except Exception:
        pass


def _start_dashboard(fn, port: int = 5000):
    try:
        from web.dashboard import start_dashboard
        start_dashboard(fn, port=port)
        logger.info(f"🌐 Web dashboard: http://localhost:{port}")
    except Exception as e:
        logger.warning(f"Dashboard start failed: {e}")


def _apply_gpu_config(cfg: dict):
    try:
        from ui.gpu_settings_panel import _patch_image_gen
        _patch_image_gen(cfg)
        logger.info(f"🎨 Image gen device: {cfg.get('image_device','auto')}")
    except Exception as e:
        logger.debug(f"GPU config skipped: {e}")


def _patch_memory_logging():
    """Monkey-patch core/memory.log_conversation to also write to cross-session history."""
    try:
        import core.memory as mem
        from core.memory_enhanced import log_turn

        _orig = mem.log_conversation

        def _patched(user: str, aura: str, meta=None):
            _orig(user, aura, meta)
            log_turn(user, aura, meta)

        mem.log_conversation = _patched
        logger.info("🧠 Cross-session memory logging active")
    except Exception as e:
        logger.warning(f"Memory patch failed: {e}")
