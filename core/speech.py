import json
import time
import logging
import pyaudio
from config import VOSK_MODEL_PATH, SAMPLE_RATE, CHUNK_SIZE, LISTEN_TIMEOUT

logger = logging.getLogger(__name__)

try:
    import webrtcvad
    webrtcvad_available = True
except ImportError:
    webrtcvad_available = False
    logger.warning("webrtcvad not installed — falling back to energy-based VAD")

# FIX: Previously this loaded at module level with no error handling.
# If the model path didn't exist, importing speech.py would crash the entire app.
# Now we load lazily on first use and fail gracefully.
_vosk_model = None

def _get_vosk_model():
    """Lazy-load the Vosk model. Returns None if unavailable."""
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model

    try:
        from vosk import Model
        _vosk_model = Model(VOSK_MODEL_PATH)
        logger.info("Vosk model loaded successfully")
        return _vosk_model
    except FileNotFoundError:
        logger.error(
            f"Vosk model not found at: {VOSK_MODEL_PATH}\n"
            f"Download a model from https://alphacephei.com/vosk/models and update VOSK_MODEL_PATH in your .env"
        )
        return None
    except Exception as e:
        logger.error(f"Vosk model failed to load: {e}")
        return None


def listen(timeout=LISTEN_TIMEOUT):
    """
    Listen for speech and return (text, success).
    Returns ("", False) on any failure — never raises.
    """
    model = _get_vosk_model()
    if model is None:
        logger.error("Speech recognition unavailable — Vosk model not loaded")
        return "", False

    start = time.time()
    stream = None
    audio = None

    try:
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        vad = webrtcvad.Vad(3) if webrtcvad_available else None

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        silence_start = None
        has_spoken = False

        while time.time() - start < timeout:
            if stream.get_read_available() < CHUNK_SIZE:
                time.sleep(0.01)
                continue

            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Voice activity detection
            if vad:
                try:
                    is_speech = vad.is_speech(data, SAMPLE_RATE)
                except Exception:
                    is_speech = False
            else:
                # Simple energy-based detection fallback
                energy = sum(
                    abs(int.from_bytes(data[i:i+2], 'little', signed=True))
                    for i in range(0, len(data), 2)
                ) / (len(data) // 2)
                is_speech = energy > 1000

            if is_speech:
                has_spoken = True
                silence_start = None
            elif has_spoken and not silence_start:
                silence_start = time.time()
            elif has_spoken and silence_start and time.time() - silence_start > 1.5:
                break  # End of utterance detected

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                txt = result.get("text", "").strip()
                if txt:
                    return txt, True

        # Grab any remaining partial result
        txt = json.loads(rec.FinalResult()).get("text", "").strip()
        return txt, bool(txt)

    except OSError as e:
        logger.error(f"Audio device error: {e}")
        return "", False
    except Exception as e:
        logger.error(f"Speech listen error: {e}")
        return "", False
    finally:
        # Always clean up audio resources — even on exception
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if audio is not None:
                audio.terminate()
        except Exception:
            pass
