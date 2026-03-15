import os
import time
import re
import subprocess
import logging
import threading
from config import PIPER_PATH, PIPER_MODEL

logger = logging.getLogger(__name__)

# ── pygame init ───────────────────────────────────────────────────────────────
try:
    import winsound
    winsound_available = True
except ImportError:
    winsound_available = False
    logger.warning("winsound not available (non-Windows system)")

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame_available = True
    logger.debug("pygame mixer initialised")
except ImportError:
    pygame_available = False
    logger.warning("pygame not installed — audio fallback unavailable. Run: pip install pygame")
except Exception as e:
    pygame_available = False
    logger.warning(f"pygame mixer init failed: {e}")


# ── Interruptible TTS wrapper ─────────────────────────────────────────────────

class InterruptibleTTS:
    def __init__(self):
        self.interrupted = False
        self.is_speaking = False

    def speak(self, text: str, cfg: dict) -> bool:
        self.interrupted = False
        self.is_speaking = True
        try:
            if text.strip() and cfg.get('voice_enabled', True):
                result = text_to_speech(text)
            else:
                result = False
        except Exception as e:
            logger.error(f"InterruptibleTTS.speak error: {e}")
            result = False
        finally:
            self.is_speaking = False
        return result

    def interrupt(self):
        self.interrupted = True
        self.is_speaking = False
        if pygame_available:
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except Exception as e:
                logger.warning(f"pygame interrupt failed: {e}")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt[:500]
    txt = re.sub(r'[\U0001F600-\U0001F64F]+', '', txt, flags=re.UNICODE)
    txt = ' '.join(txt.split())
    replacements = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'", '\u2026': '...'}
    for old, new in replacements.items():
        txt = txt.replace(old, new)
    return ''.join(c for c in txt if ord(c) < 256 and (c.isprintable() or c.isspace())).strip()


# ── TTS via Piper ─────────────────────────────────────────────────────────────

def text_to_speech(txt: str) -> bool:
    if not txt.strip():
        return False

    txt = clean_text(txt)
    if not txt.strip():
        return False

    out = f"output_{threading.current_thread().ident}.wav"
    try:
        if not os.path.exists(PIPER_PATH):
            logger.error(f"Piper executable not found at: {PIPER_PATH}  — update PIPER_PATH in your .env")
            return False

        proc = subprocess.Popen(
            [PIPER_PATH, '--model', PIPER_MODEL, '--output_file', out],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        _, stderr = proc.communicate(input=txt, timeout=30)

        if proc.returncode != 0:
            logger.error(f"Piper exited with code {proc.returncode}: {stderr.strip()}")
            return False

        if not os.path.exists(out):
            logger.error("Piper ran successfully but output WAV file was not created")
            return False

        return play_audio(out)

    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out after 30s")
        try:
            proc.kill()
        except Exception:
            pass
        return False
    except FileNotFoundError:
        logger.error(f"Piper executable not found at: {PIPER_PATH}")
        return False
    except Exception as e:
        logger.error(f"text_to_speech error: {e}")
        return False
    finally:
        # Retry deletion — winsound keeps the file handle open briefly after
        # returning, causing WinError 32 if we delete too quickly
        if os.path.exists(out):
            for attempt in range(8):
                time.sleep(0.4)
                try:
                    os.remove(out)
                    break
                except PermissionError:
                    if attempt == 7:
                        logger.warning(
                            f"Could not remove temp WAV {out} after 8 attempts "
                            f"— will be cleaned up on next run")
                except Exception as e:
                    logger.warning(f"Could not remove temp WAV {out}: {e}")
                    break


# ── Audio playback ────────────────────────────────────────────────────────────

def play_audio(fp: str) -> bool:
    if not os.path.exists(fp):
        logger.error(f"play_audio: file not found: {fp}")
        return False

    # Try winsound first (Windows only, zero dependencies)
    if winsound_available:
        try:
            winsound.PlaySound(fp, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            return True
        except Exception as e:
            logger.warning(f"winsound playback failed: {e} — trying pygame")

    # Try pygame fallback
    if pygame_available:
        try:
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            # Explicitly unload so pygame releases the file handle
            # Without this, the file stays locked and can't be deleted
            pygame.mixer.music.unload()
            return True
        except Exception as e:
            logger.error(f"pygame playback failed: {e}")
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            return False

    logger.error("No audio playback method available")
    return False


def cleanup_stale_wavs():
    """
    Delete any leftover output_*.wav files from previous runs.
    Call this once at startup so old locked files don't accumulate.
    """
    import glob
    for fp in glob.glob("output_*.wav"):
        try:
            os.remove(fp)
            logger.debug(f"Cleaned up stale WAV: {fp}")
        except Exception:
            pass  # still locked somehow — leave it
