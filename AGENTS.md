# Sabily Engineering Instructions

## Project

Sabily is a local-first Arabic short-video generation system.

Input:
- Long-form video URL

Output:
- 9:16 short clips
- MP4 files
- JSON metadata
- Local web dashboard

## Core pipeline

download
→ audio extraction
→ transcription
→ candidate scoring
→ LLM ranking/metadata
→ face tracking/reframing
→ subtitle generation
→ FFmpeg rendering
→ output validation

## Core directories

app/
  Core application and pipeline

app/pipeline/
  Processing stages

scripts/
  Diagnostics, gates and CLI utilities

tests/
  Automated tests

web/
  Dashboard/UI

assets/
  Fonts and correction dictionaries

## Mandatory principles

1. Do not rewrite working systems without evidence.
2. Preserve existing behavior unless the change explicitly requires otherwise.
3. Prefer small isolated changes.
4. Every bug fix must include a regression test when practical.
5. Every new feature must include tests.
6. Never expose secrets.
7. Never commit .env.
8. Never remove existing tests merely to make the suite pass.
9. Never disable a failing test without documenting why.
10. Never hide errors with broad exception handling.
11. Preserve Arabic RTL behavior.
12. Preserve subtitle timing correctness.
13. Preserve GPU memory safety.
14. Preserve CPU fallback behavior.
15. Preserve NVENC → x264 fallback.
16. Preserve LLM provider fallback behavior.

## Required verification

After modifications run:

python -m pytest tests -q

python -m scripts.doctor

python -m scripts.gate_arabic

python -m scripts.gate_vram

Also run targeted tests relevant to changed modules.

## Quality gates

A change is NOT complete until:

- tests pass
- diagnostics pass
- no obvious regressions exist
- changed behavior is manually reviewed
- generated outputs are validated when applicable

## Git discipline

Before major changes:

git status
git diff

Never destroy unrelated user changes.

Do not reset or checkout user changes without explicit permission.

## Completion rule

Do not report "complete" merely because code was edited.

"Complete" means:
1. implemented
2. tested
3. reviewed
4. regression checked
5. final status reported