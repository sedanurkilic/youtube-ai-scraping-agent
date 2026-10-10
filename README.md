# YouTube AI Scraping Agent

Given a YouTube song URL, the CLI finds the artist, asks an LLM for the artist's first
studio album and its tracks in original release order, scrapes the lyrics from
genius.com, computes per-song metrics, and finally produces an embedding-based hash.

```
python3 Seda_Nur_Kilic_YouTube_AI.py "<youtube_url>" json   # prints the analysis JSON
python3 Seda_Nur_Kilic_YouTube_AI.py "<youtube_url>" hash   # prints the final MD5 hash
```

## Pipeline

```
YouTube URL
  1. Apify  streamers/youtube-scraper      -> artist (channel/track metadata)
  2. OpenRouter openai/gpt-5.2             -> first album + tracks (strict JSON schema,
                                              temperature 0, seed 42)
  3. Apify  apify/web-scraper              -> Genius lyrics (slug guess, title-verified;
            (fallback: apify/google-search-scraper for candidate URLs)
  4. Local metrics                         -> chars, words, tiktoken cl100k_base tokens,
                                              tokens_per_word, MD5 of lyrics
  5. JSON: artist, album_name, songs[], total_tokens_all_songs, avg_tokens_per_song
  6. Token counts joined by "," -> nomic-ai/nomic-embed-text-v1.5 (local, 768 floats)
     -> each float f"{x:.10f}", comma-joined, no spaces -> MD5 = final hash
```

## Setup

Requires Python 3.12 (developed and tested on macOS arm64, CPU only).

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the two keys
```

`.env` (only these two variables are used):

```
APIFY_API_KEY=...
OPENROUTER_API_KEY=...
```

**One-time Apify step:** the first run of `apify/web-scraper` may fail with
`403 full-permission-actor-not-approved`. The error message contains an approval URL;
open it while logged in to Apify and approve the actor once. Then re-run.

**First run** downloads the Nomic model (~550 MB) from Hugging Face.

## Behaviour and options

- Model: `openai/gpt-5.2` by default (`OPENROUTER_MODEL` overrides it for experiments).
  `OPENROUTER_MAX_TOKENS` (default 3000) caps the completion size.
- Errors are printed as `[ERROR] <step>: <message>` (stderr) and the exit code is 1.
- Progress goes to stderr; stdout contains only the JSON or the hash.
- `.cache/` (git-ignored, contains no lyrics) stores the LLM album answer and the final
  analysis per video so repeat runs are fast and stable. `--no-cache` bypasses it.

## Determinism

- The LLM call uses strict `json_schema`, temperature 0 and a fixed seed; the prompt
  tells the model to exclude any track it is unsure about, and the answer is validated.
- Two live runs for the same video (Genius re-scraped, analysis cache deleted) produced
  byte-identical JSON. The hash path is stable across runs on the same machine.
- Embedding: model revision pinned to `e9b6763023c676ca8431644204f50c2b100d9aab`,
  CPU, float32, no text prefix, no post-processing.

Example (video `rSaC-YbSDpo`, Madonna - Borderline): album "Madonna", 8 tracks,
token counts `676,646,666,468,627,670,604,948`, total 5305, average 663.12,
hash `6a84d8020f1a875ea46463501ebc5919` (on the development machine).

## Design decisions

- **Lyrics text:** the text of all `[data-lyrics-container="true"]` elements, joined with
  `"\n"`, outer whitespace stripped. Section tags like `[Chorus]` are kept.
- **Words:** `len(text.split())`. **Tokens:** `cl100k_base`, special-token strings are
  treated as ordinary text.
- **Rounding:** Python `round(x, 2)`. The average 663.125 becomes 663.12 (round-half-even
  on the binary value); half-up would give 663.13.
- **Page verification:** a guessed Genius URL is accepted only if the page title contains
  the artist and track and no variant words (remix, live, translation, ...) that are not
  in the track name. Feature credits such as "(feat. X)" are stripped from LLM titles.
- **stdout hygiene:** the Nomic loader prints to stdout, so the pipeline runs with stdout
  redirected to stderr and only the final result is printed to the real stdout.

## Known limitations (honest notes)

- **The reference hash in the assessment text was not reproduced.** Six pre-registered
  embedding variants (2 prefixes x 3 post-processings) were tried on the reference token
  string; none matched. The simplest literal setup (no prefix, native output) is used and
  its result is documented here as unverified against the reference. Likewise, the
  example lyrics metrics for "Holiday" (1482 chars / 264 words) differ from what the
  current Genius page yields (2380 / 414), so the reference was probably made from a
  different source text.
- Hashes may differ across CPUs/PyTorch builds because float32 results can differ in the
  last digits; the pinned versions in `requirements.txt` are the tested ones.
- `trust_remote_code=True` also fetches modeling code from a second repository
  (`nomic-bert-2048`) which is not pinned.
- The Google-search fallback for Genius URLs is implemented but has not succeeded in a live
  run, because the slug guess worked for every tested track.
- The LLM can still be wrong about what counts as an artist's "first album"; the golden
  checks (Madonna, Rihanna, Adele) passed, other artists are untested.

## Tests

```bash
python -m unittest discover -s tests -v          # 100 tests (3 model tests are opt-in), no network
RUN_MODEL_TESTS=1 python -m unittest discover -s tests -v   # also loads the Nomic model
```

Tests use invented texts only (no copyrighted lyrics).
