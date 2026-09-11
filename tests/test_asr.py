"""Local ASR accuracy knobs — no model, no GPU, no ffmpeg."""

import json

from app.config import settings
from app.pipeline import transcribe


def _set(name, value):
    orig = getattr(settings, name)
    object.__setattr__(settings, name, value)
    return orig


def _restore(name, orig):
    object.__setattr__(settings, name, orig)


def test_effective_asr_balanced_is_stock():
    o1, o2, o3 = (_set("whisper_accuracy", "balanced"),
                  _set("whisper_condition", False),
                  _set("beam_size", 5))
    try:
        import os
        old = os.environ.pop("WHISPER_MODEL", None)
        try:
            model, beam, cond = transcribe._effective_asr()
            assert model == "large-v3-turbo"
            assert beam == 5 and cond is False
        finally:
            if old is not None:
                os.environ["WHISPER_MODEL"] = old
    finally:
        _restore("whisper_accuracy", o1)
        _restore("whisper_condition", o2)
        _restore("beam_size", o3)


def test_effective_asr_accurate_and_fast():
    import os
    old = os.environ.pop("WHISPER_MODEL", None)
    o1, o2 = _set("whisper_accuracy", "accurate"), _set("beam_size", 5)
    try:
        model, beam, cond = transcribe._effective_asr()
        assert model == "large-v3" and cond is True
        o3 = _set("whisper_accuracy", "fast")
        try:
            model, beam, cond = transcribe._effective_asr()
            assert model == "small" and beam == 1 and cond is False
        finally:
            _restore("whisper_accuracy", o3)
    finally:
        _restore("whisper_accuracy", o1)
        _restore("beam_size", o2)
        if old is not None:
            os.environ["WHISPER_MODEL"] = old


def test_effective_asr_explicit_model_wins(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    o1 = _set("whisper_accuracy", "accurate")
    try:
        assert transcribe._effective_asr()[0] == "medium"
    finally:
        _restore("whisper_accuracy", o1)


def test_hotwords_auto_off_custom(tmp_path):
    terms = tmp_path / "terms.json"
    terms.write_text(json.dumps({"replace": {"كيفريم": "Keyframe",
                                             "زوم إين": "Zoom in"},
                                         "split": []}), encoding="utf-8")
    o1, o2 = _set("terms_file", terms), _set("whisper_hotwords", "auto")
    try:
        hw = transcribe._hotwords()
        assert hw and "Keyframe" in hw and "Zoom" in hw
        o3 = _set("whisper_hotwords", "off")
        try:
            assert transcribe._hotwords() is None
        finally:
            _restore("whisper_hotwords", o3)
        o4 = _set("whisper_hotwords", "مونتاج بريمير")
        try:
            assert transcribe._hotwords() == "مونتاج بريمير"
        finally:
            _restore("whisper_hotwords", o4)
    finally:
        _restore("terms_file", o1)
        _restore("whisper_hotwords", o2)


def test_transcribe_kwargs_shape():
    kw = transcribe._transcribe_kwargs()
    assert kw["word_timestamps"] is True and kw["vad_filter"] is True
    assert kw["vad_parameters"]["min_silence_duration_ms"] == settings.vad_min_silence
    assert kw["beam_size"] >= 1 and kw["best_of"] == kw["beam_size"]
    for k in ("compression_ratio_threshold", "log_prob_threshold",
              "no_speech_threshold", "condition_on_previous_text"):
        assert k in kw


def test_stale_cache_retranscribes(tmp_path, monkeypatch):
    from app.pipeline.transcribe import Word

    cache = tmp_path / "transcript.json"
    cache.write_text(json.dumps({"fingerprint": {"model": "tiny"},
                                 "words": [{"start": 0.0, "end": 1.0,
                                            "text": "قديم", "conf": 1.0}]}),
                     encoding="utf-8")
    monkeypatch.setattr(transcribe, "_transcribe",
                        lambda audio, device, compute, on_progress: [
                            Word(start=0.0, end=1.0, text="جديد")])
    words = transcribe.run(tmp_path / "a.wav", cache=cache)
    assert [w.text for w in words] == ["جديد"]
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["fingerprint"] == transcribe._asr_fingerprint()


def test_matching_cache_hit_bypasses_model(tmp_path, monkeypatch):
    cache = tmp_path / "transcript.json"
    words = [{"start": 0.0, "end": 1.0, "text": "ok", "conf": 1.0}]
    cache.write_text(json.dumps({"fingerprint": transcribe._asr_fingerprint(),
                                 "words": words}), encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("model must not run on cache hit")

    monkeypatch.setattr(transcribe, "_transcribe", _boom)
    assert [w.text for w in transcribe.run(tmp_path / "a.wav", cache=cache)] == ["ok"]
