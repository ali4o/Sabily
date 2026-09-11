"""Second feature wave — scoring/audio, LLM polish, subtitle styles,
bumpers/zoom, cancel/cleanup/zip/audio-preview. No GPU, no network."""

from app import jobs
from app.config import settings
from app.pipeline.transcribe import Word


def _set(name, value):
    orig = getattr(settings, name)
    object.__setattr__(settings, name, value)
    return orig


def _restore(name, orig):
    object.__setattr__(settings, name, orig)


def _words(text="مرحبا بالجميع. كيف الحال اليوم؟"):
    toks, out, t = text.split(), [], 0.0
    for tok in toks:
        out.append(Word(start=t, end=t + 0.4, text=tok))
        t += 0.5
    return out


# --- scoring: audio penalty is zero when clean/absent --------------------

def test_audio_score_neutral_and_penalty():
    from app.pipeline.score import _audio_score, score_candidate, build_sentences

    assert _audio_score(None) == 1.0
    assert _audio_score({"silence": 0.0, "clipped": 0.0}) == 1.0
    assert _audio_score({"silence": 0.9, "clipped": 0.0}) < 0.5
    assert _audio_score({"silence": 0.0, "clipped": 1.0}) == 0.5
    sents = build_sentences(_words())
    base = score_candidate(sents, {}).score
    same = score_candidate(sents, {}, {"silence": 0.0, "clipped": 0.0}).score
    assert base == same
    bad = score_candidate(sents, {}, {"silence": 1.0, "clipped": 0.0}).score
    assert bad < base


def test_diversify_drops_twins_and_fills():
    from app.pipeline.score import Candidate, diversify

    a = Candidate(10.0, 40.0, "لماذا يفشل الناس في العادات", 0.9)
    b = Candidate(12.0, 42.0, "لماذا يفشل الناس في العادات", 0.8)  # twin
    c = Candidate(200.0, 230.0, "البيئة أهم من الإرادة تماما", 0.7)
    assert [x.start for x in diversify([a, b, c], 2)] == [10.0, 200.0]
    assert diversify([a], 3) == [a]


def test_detect_turns_alternates():
    from app.pipeline.score import detect_turns

    words = [Word(start=0.0, end=0.4, text="أهلا"),
             Word(start=3.0, end=3.4, text="مرحبا"),
             Word(start=3.6, end=4.0, text="بك")]
    turns = detect_turns(words, turn_sec=1.5)
    assert [t["speaker"] for t in turns] == [1, 2]
    assert turns[0]["end"] == 0.4 and turns[1]["start"] == 3.0


def test_audio_q_stats_silent_wav(tmp_path):
    import wave

    from app.pipeline import audio_q

    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    audio = audio_q.load(p)
    assert audio is not None
    stats = audio_q.stats_from(audio, 0.0, 1.0)
    assert stats["silence"] > 0.9
    assert audio_q.load(tmp_path / "missing.wav") is None
    assert audio_q.stats_from(None, 0.0, 1.0) is None


# --- llm polish / translate fallbacks -------------------------------------

def test_polish_disabled_and_no_provider(monkeypatch):
    from app.pipeline import llm

    o1 = _set("transcript_polish", False)
    try:
        assert llm.polish_text("نص") == "نص"
    finally:
        _restore("transcript_polish", o1)
    monkeypatch.setattr(llm, "get_provider", lambda: None)
    assert llm.polish_text("نص") == "نص"
    assert llm.translate_lines(["أهلا", "تمام"]) == ["أهلا", "تمام"]


def test_translate_bad_shape_falls_back(monkeypatch):
    from app.pipeline import llm

    class P:
        def complete(self, s, u):
            return '{"nope": 1}'

    monkeypatch.setattr(llm, "get_provider", lambda: P())
    o1 = _set("subtitle_bilingual", True)
    try:
        assert llm.translate_lines(["أهلا"]) == ["أهلا"]
    finally:
        _restore("subtitle_bilingual", o1)


# --- subtitle styles -------------------------------------------------------

def test_subtitle_pos_alignment(tmp_path):
    from app.pipeline import subtitle

    out = tmp_path / "t.ass"
    subtitle.write_ass(_words(), 0.0, 5.0, out, lines=[
        {"start": 0.0, "end": 1.0, "text": "hi"}], pos="middle", burn=True)
    text = out.read_text(encoding="utf-8")
    assert ",2,5,60,60," in text  # Sabily style alignment = 5
    assert subtitle.resolve_sub_align("bogus") == 2


def test_subtitle_karaoke_tags(tmp_path):
    from app.pipeline import subtitle

    out = tmp_path / "t.ass"
    subtitle.write_ass(_words(), 0.0, 5.0, out, style="karaoke", burn=True,
                       show_brand=False, show_source=False)
    text = out.read_text(encoding="utf-8")
    assert "\\kf" in text


def test_subtitle_bilingual_second_line(tmp_path):
    from app.pipeline import subtitle

    out = tmp_path / "t.ass"
    subtitle.write_ass(_words(), 0.0, 5.0, out,
                       lines=[{"start": 0.0, "end": 1.0, "text": "مرحبا"}],
                       en_lines=["hello"], burn=True,
                       show_brand=False, show_source=False)
    text = out.read_text(encoding="utf-8")
    assert "مرحبا\\Nhello" in text


def test_label_turns_prefix():
    from app.pipeline.subtitle import label_turns

    rows = [{"start": 0.0, "end": 2.0, "text": "أهلا"},
            {"start": 2.0, "end": 4.0, "text": "مرحبا"}]
    turns = [{"start": 0.0, "end": 2.0, "speaker": 1},
             {"start": 2.0, "end": 4.0, "speaker": 2}]
    out = label_turns(rows, turns, 0.0)
    assert out[0]["text"] == "أهلا"
    assert out[1]["text"].startswith("المتحدث 2:")


def test_write_card_ass(tmp_path):
    from app.pipeline import subtitle

    out = subtitle.write_card_ass("سبيلي", 720, 1280, tmp_path / "c.ass")
    text = out.read_text(encoding="utf-8")
    assert "سبيلي" in text and ",2,5," in text


# --- render: zoom + bumpers -------------------------------------------------

def test_zoom_filter_shape():
    from app.pipeline.render import zoom_filter

    assert zoom_filter([]) == "" and zoom_filter([0.0]) == ""
    expr = zoom_filter([0.0, 2.5, 5.0])
    assert "lt(t,2.50)" in expr and "1.06" in expr


def test_build_bumper_and_concat(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from app.pipeline import render

    def fake_run(cmd, **kw):
        Path = __import__("pathlib").Path
        Path(cmd[-1]).write_bytes(b"fake-mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    card = render.build_bumper("intro", 720, 1280, "سبيلي", tmp_path, False)
    assert card and card.exists()
    assert render.concat_parts([card, card], tmp_path / "out.mp4") is True


def test_render_clip_zoom_in_vf(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from app.pipeline import render

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    monkeypatch.setattr(render, "verify_output", lambda *a, **k: None)
    orig_zoom = settings.zoom_punch
    object.__setattr__(settings, "zoom_punch", True)
    try:
        crop = {"w": 608, "h": 1080, "x_expr": "if(lt(t,1.0),1,2)", "y": 0,
                "mode": "dynamic", "zoom_times": [0.0, 2.0]}
        render.render_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", 0, 5,
                           crop, None, target_size=(1080, 1920),
                           show_logo=False, burn_subtitles=False,
                           zoom_times=[0.0, 2.0])
        vf = calls[-1][calls[-1].index("-vf") + 1]
        assert "iw*(" in vf and "crop=1080:1920" in vf
    finally:
        object.__setattr__(settings, "zoom_punch", orig_zoom)


# --- jobs: render log, cleanup, cancel guard --------------------------------

def test_render_log_roundtrip(tmp_path):
    from app import jobs

    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jobs.log_render("abcdef012345", "FHD", 30.0, 5000, 12.5)
        rows = jobs.list_renders()
        assert len(rows) == 1 and rows[0]["quality"] == "FHD"
        assert rows[0]["mb"] == 12.5
    finally:
        object.__setattr__(settings, "db_path", orig)


def test_cleanup_old_sources(tmp_path):
    import time

    from app import jobs

    orig_db, orig_work = settings.db_path, settings.work_dir
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    object.__setattr__(settings, "work_dir", tmp_path / "work")
    try:
        jobs.init_db()
        old = jobs.create_job("https://example.com/o", {})
        jobs.update_job(old, status="done")
        with jobs.connect() as conn:
            conn.execute("UPDATE jobs SET updated_at=? WHERE id=?",
                         (time.time() - 10 * 86400, old))
            conn.commit()
        new = jobs.create_job("https://example.com/n", {})
        jobs.update_job(new, status="done")
        (tmp_path / "work" / old).mkdir(parents=True)
        (tmp_path / "work" / new).mkdir(parents=True)
        assert jobs.cleanup_old_sources(7) == 1
        assert not (tmp_path / "work" / old).exists()
        assert (tmp_path / "work" / new).exists()
        assert jobs.cleanup_old_sources(0) == 0
    finally:
        object.__setattr__(settings, "db_path", orig_db)
        object.__setattr__(settings, "work_dir", orig_work)


def test_runner_cancel_guard(monkeypatch):
    from app import jobs
    from app.pipeline import runner

    monkeypatch.setattr(jobs, "get_job",
                        lambda jid: {"id": jid, "status": "cancelled"})
    # must return before touching the network
    assert runner.process("abcdef012345") is None


# --- main endpoints ----------------------------------------------------------

def test_cancel_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jid = jobs.create_job("https://example.com/v", {})
        c = TestClient(app, raise_server_exceptions=False)
        assert c.post("/api/jobs/000000000000/cancel").status_code == 404
        r = c.post(f"/api/jobs/{jid}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        r = c.post(f"/api/jobs/{jid}/cancel")
        assert r.json()["already"] == "cancelled"
    finally:
        object.__setattr__(settings, "db_path", orig)


def test_audio_and_zip_missing(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        c = TestClient(app, raise_server_exceptions=False)
        assert c.get("/api/clips/0123456789ab/audio").status_code == 404
        jid = jobs.create_job("https://example.com/v", {})
        assert c.get(f"/api/jobs/{jid}/zip").status_code == 404
        assert c.get("/api/jobs/zzzz/zip").status_code in (400, 404)
    finally:
        object.__setattr__(settings, "db_path", orig)


def test_stats_has_renders(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    orig = settings.db_path
    object.__setattr__(settings, "db_path", tmp_path / "t.db")
    try:
        jobs.init_db()
        jobs.log_render("abcdef012345", "QHD", 20.0, 9000, 30.0)
        r = TestClient(app, raise_server_exceptions=False).get("/api/stats")
        assert r.status_code == 200
        assert r.json()["renders"][0]["quality"] == "QHD"
    finally:
        object.__setattr__(settings, "db_path", orig)
