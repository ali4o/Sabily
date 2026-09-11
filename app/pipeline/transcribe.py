"""Stage 2 — ASR with faster-whisper.

The model is loaded, used, and freed inside one function on purpose. Nothing
else in the pipeline may touch the GPU while this runs.
"""

import gc
import json
import logging
import os
import site
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from app.config import settings

log = logging.getLogger("sabily.asr")


@dataclass
class Word:
    start: float
    end: float
    text: str
    conf: float = 1.0     # 0..1, derived from the segment's avg_logprob


def _add_cuda_dll_dirs() -> list[str]:
    """Windows: make the pip-installed CUDA libraries loadable.

    CTranslate2 links cuBLAS and cuDNN at runtime and searches PATH only.
    The nvidia-*-cu12 wheels drop their DLLs inside site-packages instead, so
    without this the GPU path dies with "cublas64_12.dll is not found".
    """
    if sys.platform != "win32":
        return []
    added = []
    for mod in ("cublas", "cudnn", "cuda_runtime"):
        for base in site.getsitepackages() + [site.getusersitepackages()]:
            d = Path(base) / "nvidia" / mod / "bin"
            if d.exists():
                try:
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = f"{d}{os.pathsep}" + os.environ.get("PATH", "")
                    added.append(str(d))
                except OSError:
                    pass
                break
    return added


def cuda_libs_present() -> bool:
    """True when the CUDA runtime libraries CTranslate2 needs are reachable."""
    if sys.platform != "win32":
        return True
    _add_cuda_dll_dirs()
    return any(
        (Path(b) / "nvidia" / "cublas" / "bin").exists()
        for b in site.getsitepackages() + [site.getusersitepackages()]
    ) or any(
        Path(d).joinpath("cublas64_12.dll").exists()
        for d in os.environ.get("PATH", "").split(os.pathsep) if d
    )


def _is_cuda_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("cublas", "cudnn", "cuda", "gpu", "device"))


def _release() -> None:
    gc.collect()
    try:
        import torch  # optional; ctranslate2 does not require it

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# accuracy presets: an explicit WHISPER_MODEL always wins; otherwise the
# preset picks the model. Threshold/VAD/prompt knobs below are
# faster-whisper defaults, so `balanced` transcribes exactly like before.
PRESET_MODELS = {"fast": "small", "balanced": "large-v3-turbo", "accurate": "large-v3"}


def _effective_asr() -> tuple[str, int, bool]:
    """(model, beam, condition_on_previous_text) for this job."""
    acc = (settings.whisper_accuracy or "balanced").strip().lower()
    if acc not in PRESET_MODELS:
        acc = "balanced"
    model = os.getenv("WHISPER_MODEL") or settings.whisper_model
    if not os.getenv("WHISPER_MODEL") and settings.whisper_model == "large-v3-turbo":
        # stock default follows the preset; any configured model is kept
        model = PRESET_MODELS[acc]
    beam = settings.beam_size
    if acc == "fast":
        beam = min(beam, 1)
    condition = settings.whisper_condition or acc == "accurate"
    return model, max(1, beam), condition


def _hotwords() -> str | None:
    """Bias the decoder toward the user's glossary terms (local, free).

    WHISPER_HOTWORDS=off disables it; any other non-auto string is used
    verbatim; `auto` collects Latin-script glossary targets (technical
    terms Whisper otherwise guesses at, e.g. Keyframe).
    """
    mode = (settings.whisper_hotwords or "auto").strip()
    if not mode or mode.lower() == "off":
        return None
    if mode.lower() != "auto":
        return mode
    try:
        data = json.loads(settings.terms_file.read_text(encoding="utf-8"))
        latin: list[str] = []
        for dst in list(data.get("replace", {}).values()):
            for tok in str(dst).split():
                clean = "".join(ch for ch in tok
                                if ch.isascii() and (ch.isalnum() or ch in "-_"))
                if len(clean) >= 3 and clean not in latin:
                    latin.append(clean)
                if len(latin) >= 20:
                    break
            if len(latin) >= 20:
                break
        return " ".join(latin) or None
    except Exception:  # noqa: BLE001 - glossary must never break ASR
        return None


def _transcribe_kwargs() -> dict:
    """Pure helper (unit-testable): every decoding knob in one place."""
    _, beam, condition = _effective_asr()
    vad: dict = {"min_silence_duration_ms": settings.vad_min_silence,
                 "speech_pad_ms": settings.vad_speech_pad}
    if settings.vad_max_speech > 0:
        vad["max_speech_duration_s"] = settings.vad_max_speech
    return {
        "language": settings.whisper_lang or None,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": vad,
        "beam_size": beam,
        "best_of": beam,
        "condition_on_previous_text": condition,
        "initial_prompt": settings.whisper_prompt or None,
        "hotwords": _hotwords(),
        "compression_ratio_threshold": settings.whisper_compression,
        "log_prob_threshold": settings.whisper_logprob,
        "no_speech_threshold": settings.whisper_no_speech,
    }


def _asr_fingerprint() -> dict:
    """Settings affecting output; stored with the cache to detect staleness."""
    model, beam, condition = _effective_asr()
    return {"model": model, "beam": beam, "condition": condition,
            "lang": settings.whisper_lang, "kwargs": _transcribe_kwargs()}


def _transcribe(
    audio: Path,
    device: str,
    compute: str,
    on_progress: Callable[[float], None] | None,
) -> list[Word]:
    """One attempt on one device. The model is freed before returning."""
    from faster_whisper import WhisperModel

    model_name, _, _ = _effective_asr()
    log.info("loading %s on %s (%s)", model_name, device, compute)
    model = WhisperModel(model_name, device=device, compute_type=compute)
    words: list[Word] = []
    try:
        segments, info = model.transcribe(str(audio), **_transcribe_kwargs())
        total = float(getattr(info, "duration", 0) or 0)
        for seg in segments:
            # avg_logprob is roughly -1.0 (bad) to 0.0 (confident)
            conf = max(0.0, min(1.0, 1.0 + float(getattr(seg, "avg_logprob", 0.0) or 0.0)))
            for w in (seg.words or []):
                text = (w.word or "").strip()
                if text:
                    words.append(Word(start=float(w.start), end=float(w.end),
                                      text=text, conf=conf))
            if on_progress and total:
                on_progress(min(1.0, seg.end / total))
    finally:
        del model
        _release()
    return words


def run(
    audio: Path,
    cache: Optional[Path] = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[Word]:
    """Transcribe to word-level timestamps. Cached to JSON so reruns are free."""
    if cache and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # new format: re-transcribe when accuracy settings changed
                if data.get("fingerprint") != _asr_fingerprint():
                    log.info("ASR settings changed since cache, re-transcribing")
                    raise ValueError("stale cache")
                return [Word(**w) for w in data.get("words", [])]
            return [Word(**w) for w in data]  # legacy list format
        except ValueError:
            try:
                cache.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        except Exception:  # noqa: BLE001 - corrupt cache must not fail the job
            log.warning("transcript cache corrupt, re-transcribing")
            try:
                cache.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass

    _add_cuda_dll_dirs()

    attempts = [(settings.whisper_device, settings.whisper_compute)]
    if settings.whisper_device != "cpu":
        attempts.append(("cpu", "int8"))

    last: Exception | None = None
    for device, compute in attempts:
        try:
            words = _transcribe(audio, device, compute, on_progress)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if device == "cpu" or not _is_cuda_error(exc):
                raise
            log.warning(
                "GPU path failed (%s) — retrying on CPU. "
                "لتشغيل الـ GPU: python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12",
                str(exc)[:150],
            )
    else:
        raise last  # type: ignore[misc]

    if cache:
        cache.write_text(
            json.dumps({"fingerprint": _asr_fingerprint(),
                        "words": [asdict(w) for w in words]},
                       ensure_ascii=False),
            encoding="utf-8",
        )
    log.info("transcribed %d words", len(words))
    return words
