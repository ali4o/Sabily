"""Gate 2 — can Whisper and the local LLM run back-to-back on 4GB VRAM?

    python -m scripts.gate_vram

Loads Whisper on a 30-second tone, releases it, then pings the LLM.
If this passes, the pipeline's stage order is safe on this machine.
"""

import subprocess
import sys
import time

from app.config import settings
from app.pipeline import llm, transcribe


def vram() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{used:.2f}/{total:.1f} GB"
    except Exception:  # noqa: BLE001
        pass
    return "n/a"


def main() -> int:
    settings.ensure_dirs()
    wav = settings.work_dir / "gate_tone.wav"
    subprocess.run(
        [settings.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
         "-ar", "16000", "-ac", "1", str(wav)],
        check=True,
    )

    print(f"VRAM before whisper : {vram()}")
    t0 = time.time()
    words = transcribe.run(wav)
    print(f"whisper done in {time.time()-t0:.1f}s ({len(words)} words)")
    print(f"VRAM after release  : {vram()}")

    provider = llm.get_provider()
    if provider is None:
        print("LLM_PROVIDER=none — skipping LLM check")
        return 0

    t0 = time.time()
    meta = llm.generate_metadata("لماذا يفشل أغلب الناس في بناء العادات الجديدة؟ السبب بسيط.")
    print(f"llm done in {time.time()-t0:.1f}s → {meta}")
    print(f"VRAM after llm      : {vram()}")
    print("PASS — الترتيب آمن على هذا الجهاز")
    return 0


if __name__ == "__main__":
    sys.exit(main())
