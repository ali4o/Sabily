"""Post-ASR cleanup.

Whisper is good but not consistent: it writes the same term in Arabic once and
in Latin the next time, and it sometimes glues two words together. Neither is
fixable inside the model, so we fix it after — with a glossary the user owns
and can edit, not rules buried in code.

assets/terms.json:
    {
      "replace": {"كيفريم": "Keyframe", "زوم إن": "zoom in"},
      "split":   ["وأنا", "ولا"]
    }
"""

import copy
import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

from app.config import settings
from app.pipeline.transcribe import Word

log = logging.getLogger("sabily.normalize")

DEFAULT_TERMS = {
    "replace": {
        "كيفريم": "Keyframe",
        "الكيفريم": "الـKeyframe",
        "كي فريم": "Keyframe",
        "زوم إين": "Zoom in",
        "زوم اوت": "Zoom out",
        "لايت موشن": "Light Motion",
        "المنتاج": "المونتاج",
        "المحشوى": "المحتوى",
        "الضفط": "الضغط",
    },
    "split": [],
}


def load_terms() -> dict:
    """Always returns a fresh copy.

    Handing back DEFAULT_TERMS itself let callers mutate the module-level dict,
    so a term learned in one job leaked into every later job in the same
    process — invisible in the file, visible in the output.
    """
    path = settings.terms_file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(DEFAULT_TERMS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return copy.deepcopy(DEFAULT_TERMS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"replace": data.get("replace", {}), "split": data.get("split", [])}
    except Exception as exc:  # noqa: BLE001 - a broken glossary must not stop a job
        log.warning("terms file unreadable (%s), using defaults", exc)
        return copy.deepcopy(DEFAULT_TERMS)


def _strip_punct(token: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)([.،؟!?…]*)$", token, re.S)
    return (m.group(1), m.group(2)) if m else (token, "")


def apply(words: Iterable[Word], terms: dict | None = None) -> list[Word]:
    """Rewrite word text in place-ish, keeping every timestamp intact.

    Multi-word glossary entries are matched across consecutive words and
    collapsed onto the first one, so timings never drift.
    """
    terms = terms or load_terms()
    replace: dict[str, str] = terms.get("replace", {})
    split_pairs: list[str] = terms.get("split", [])
    if not replace and not split_pairs:
        return list(words)

    # longest first so "زوم إين" wins over "زوم"
    phrases = sorted(replace.items(), key=lambda kv: -len(kv[0].split()))
    items = list(words)
    out: list[Word] = []
    i = 0
    while i < len(items):
        matched = False
        for phrase, target in phrases:
            parts = phrase.split()
            window = items[i : i + len(parts)]
            if len(window) < len(parts):
                continue
            bare = [_strip_punct(w.text)[0] for w in window]
            if bare == parts:
                tail = _strip_punct(window[-1].text)[1]
                out.append(Word(start=window[0].start, end=window[-1].end,
                                text=target + tail))
                i += len(parts)
                matched = True
                break
        if matched:
            continue

        w = items[i]
        bare, tail = _strip_punct(w.text)
        for glued in split_pairs:
            if bare == glued and len(glued) > 3:
                bare = f"{glued[0]} {glued[1:]}"
                break
        out.append(Word(start=w.start, end=w.end, text=bare + tail))
        i += 1

    changed = sum(1 for a, b in zip(items, out) if a.text != b.text)
    if changed:
        log.info("normalized %d tokens", changed)
    return out


def learn(old_lines: list[dict], new_lines: list[dict], limit: int = 12) -> dict[str, str]:
    """Turn the user's subtitle edits into glossary entries.

    This is the loop that makes the tool improve: a correction made once in the
    editor is applied automatically to every future video. Only short,
    unambiguous word swaps are learned — rewrites of a whole line are the
    user's phrasing, not a transcription fix, and must not become a rule.
    """
    if not old_lines or not new_lines:
        return {}
    terms = load_terms()
    replace = dict(terms.get("replace", {}))
    learned: dict[str, str] = {}

    for old, new in zip(old_lines, new_lines):
        a, b = str(old.get("text", "")).split(), str(new.get("text", "")).split()
        if not a or not b or a == b:
            continue
        if abs(len(a) - len(b)) > 2:
            continue          # a full rewrite, not a fix
        sm = difflib.SequenceMatcher(a=a, b=b)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) > 2:
                continue
            src = _strip_punct(a[i1])[0]
            dst = " ".join(_strip_punct(w)[0] for w in b[j1:j2])
            if not src or not dst or src == dst or len(src) < 3 or len(dst) > 40:
                continue
            if src in replace and replace[src] == dst:
                continue
            learned[src] = dst
            if len(learned) >= limit:
                break

    if not learned:
        return {}
    replace.update(learned)
    if len(replace) > 500:
        replace = dict(list(replace.items())[-500:])
    terms["replace"] = replace
    settings.terms_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings.terms_file.with_name(settings.terms_file.name + ".tmp")
    tmp.write_text(
        json.dumps(terms, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, settings.terms_file)
    log.info("learned %d term(s) from edits: %s", len(learned), learned)
    return learned
