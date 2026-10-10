# Decisions and calibration findings (numbers only, no lyric text)

Working log; the README summarizes it.

## Embedding (Phase 1, calibrate_embedding.py)
- Input: `395,703,470,622,561,950,712,634`; 6 pre-registered trials (2 prefixes x 3 post-processings).
- Result: **no trial reproduced the reference MD5** `4ae6acfb...`. Mismatch recorded as an open finding; it does not prove the Nomic configuration is wrong (the reference may come from a different source pipeline or hardware).
- Selected: **T1** = no prefix, native output, 768 dims, CPU, float32. Status: UNVERIFIED against the reference. T1 MD5 on the author's machine: `041ccada537fbecbde13a3936c77ecb7` (regression value only).
- Model revision pinned: `e9b6763023c676ca8431644204f50c2b100d9aab`. Known limit: `trust_remote_code` also loads code from a second repo (`nomic-bert-2048`) that is not pinned.
- Environment: macOS arm64, Python 3.12.2, torch 2.14.1, sentence-transformers 3.1.1, transformers 4.44.2, numpy 2.5.3, einops 0.8.2, huggingface_hub 0.36.2.
- Repeat runs on the same machine were bit-identical (max abs diff 0.0).

## Genius access (Phase 2, spike_genius.py)
- `apify/web-scraper` reached `https://genius.com/Madonna-holiday-lyrics` with the default Apify proxy on the first attempt (HTTP 200, title "Madonna - Holiday Lyrics | Genius Lyrics"); residential proxy was not needed.
- One-time prerequisite: approve the actor's full-permission prompt in Apify Console (otherwise HTTP 403 `full-permission-actor-not-approved`). Must be in the README.

## Lyrics policy (Phase 2, calibrate_lyrics.py)
- Raw page: 5 lyrics containers, innerText lengths `[1022, 0, 548, 0, 806]` (two empty containers).
- Selected: **P2** = innerText of the containers joined with "\n", outer whitespace trimmed, nothing inside removed or rewritten.
- P2 numbers: 2380 chars / 414 words (split) / 627 tokens. Reference (Holiday): 1482 / 264 / 395 / MD5 `1933a103...`.
- No pre-registered policy (P1-P7) and no diagnostic prefix (D1-D3) matched the reference. The gap is content volume (~+60% chars), not cleaning. Cause unknown. Not pursued further.
- Word rule: **`len(text.split())`** (regex alternative differs by 1 word: 414 vs 415).
- Rounding: Python `round(x, 2)`.

## Consequence
- The final hash depends only on per-song token counts. Holiday here is 627 tokens vs 395 in the reference, so an exact match with the assessment's example hash is not expected.

## YouTube metadata (Phase 2, spike_youtube.py)
- `streamers/youtube-scraper` (input: only `startUrls`) returned 31 fields for the assessment URL; useful ones: `title`, `channelName`, `text` (description).
- Metadata: title "Madonna - Borderline (Official Video) [HD]", channel "Madonna".
- Rule: the artist is NOT taken from the channel name alone (VEVO/Topic/third-party channels). Title + channel + first 400 chars of the description go to the LLM, which is told to use them only to identify the artist.

## Album discovery with the LLM (Phase 3, spike_llm.py)
- Model `openai/gpt-5.2` via OpenRouter, strict JSON schema, `temperature=0`, `seed=42` sent (not proof they are honored).
- **Prompt v1 failed:** returned 9 tracks, adding "Stay" after the original 8; its `notes` field confidently called it a "9-track core lineup". Several independent sources list the original 1983 standard edition as 8 tracks (Lucky Star ... Everybody). Lesson: never trust the model's own notes; validate.
- **Prompt v2** (stricter "exclude if unsure" rule + required `excluded_candidates` list + local overlap validation): 3/3 runs returned the same 8 tracks in the original order (artist, album, year, tracks identical).
- `notes` and `excluded_candidates` differ between runs and output tokens varied (381-605); only artist/album/year/tracks are used downstream.
- Caveat: v2 was written after seeing the v1 failure on this same artist (in-sample). Generalization is to be checked on other artists before the prompt is frozen.
- Transient network failures (observed once: connection closed by remote) are retried up to 3 attempts; 401/402 are not retried.
- Reference example lists Holiday first and Borderline last; the real original order differs. Our rule stays: original release order.

## Generalization check (Phase 3.2, eval_golden.py)
- Three vetted albums (`tests/golden_albums.json`): Madonna, Rihanna, Adele. Result: **all three PASS** with prompt v2 (prompt frozen after this check; no further tuning).
- Rihanna initially failed only because the model returned "(feat. J-Status)" in a title. Fix: `clean_track_title()` strips trailing feature credits ("(feat. X)", "(ft. X)", ...), applied in the pipeline and in the evaluation.
- OpenRouter credits ran out several times (HTTP 402). Added: configurable `OPENROUTER_MAX_TOKENS`, a clear 402 error message, and an album cache so a successful LLM answer is never paid for twice.

## Genius scraping hardening (Phase 4)
- A second live run ended with Apify `TIMED-OUT` after the Google fallback and lost the LLM result. Fixes: the album answer is cached before scraping; the scraper uses `waitUntil: domcontentloaded`, page-load timeout 45 s, page-function timeout 30 s; rejection reasons are logged.
- Candidate pages are guessed as `https://genius.com/{Artist}-{title}-lyrics` and accepted only if the page title matches the artist and track and contains no variant words (remix, live, translation, ...) absent from the track name. All 8 pages for the test video were accepted via the slug guess; the Google-search fallback never had to succeed live (untested path).

## Live results and determinism (Phase 4)
- Test video: `rSaC-YbSDpo` (Madonna - Borderline). Album "Madonna", 8 tracks.
- Token counts: `676,646,666,468,627,670,604,948`; total 5305; average 663.125 -> **663.12** (Python round-half-even; half-up would be 663.13).
- Final hash on the author's machine: **`6a84d8020f1a875ea46463501ebc5919`**.
- Reproducibility: with the analysis cache deleted and Genius re-scraped through Apify, the JSON output was byte-identical to the first run (`diff` empty). One earlier attempt produced an empty file because of a transient Apify failure; the error was hidden by `2>/dev/null`, a reminder that stderr must be visible when testing.
- Not verified live: a fully cache-free end-to-end run with `gpt-5.2` from a clean environment (no OpenRouter credits left at that time). The album answer used in the live runs was produced by `gpt-5.2` with prompt v2 and confirmed by the 3-run stability test.
- Test suite: 100 unit tests (mocked network); 3 model-dependent tests are skipped unless RUN_MODEL_TESTS=1; model-dependent tests are opt-in with `RUN_MODEL_TESTS=1`.