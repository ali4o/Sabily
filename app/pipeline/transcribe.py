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


def _transcribe(
    audio: Path,
    device: str,
    compute: str,
    on_progress: Callable[[float], None] | None,
) -> list[Word]:
    """One attempt on one device. The model is freed before returning."""
    from faster_whisper import WhisperModel

    log.info("loading %s on %s (%s)", settings.whisper_model, device, compute)
    model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
    words: list[Word] = []
    try:
        segments, info = model.transcribe(
            str(audio),
            language=settings.whisper_lang or None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=1,          # 4GB VRAM: greedy is enough and much lighter
            condition_on_previous_text=False,
        )
        total = float(getattr(info, "duration", 0) or 0)
        for seg in segments:
            for w in (seg.words or []):
                text = (w.word or "").strip()
                if text:
                    words.append(Word(start=float(w.start), end=float(w.end), text=text))
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
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [Word(**w) for w in data]

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
            json.dumps([asdict(w) for w in words], ensure_ascii=False),
            encoding="utf-8",
        )
    log.info("transcribed %d words", len(words))
    return words
