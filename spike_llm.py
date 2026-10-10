#!/usr/bin/env python3
"""
Phase 3 / micro-step 3.1 - OpenRouter album-discovery spike (development tool, NOT
part of the submission).

Question: given the YouTube metadata, can ONE OpenRouter call return the artist, the
first studio album and its original-release-order tracklist as strict JSON - and is
the answer identical across repeated calls?

* Input is read from spike_output/youtube_raw.json (no Apify credits).
* Only OpenRouter; OPENROUTER_API_KEY from the environment or .env.
* Default model: openai/gpt-5.2 (the model the assessment will be graded with).
* Runs the identical request N times (default 3) and compares the answers.
* Saves everything to spike_output/llm_runs.json (git-ignored).

Usage:
    python3 spike_llm.py
    python3 spike_llm.py --runs 3 --model openai/gpt-5.2
"""

import argparse
import json
import os
import sys
import time

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
YOUTUBE_PATH = os.path.join("spike_output", "youtube_raw.json")
OUT_PATH = os.path.join("spike_output", "llm_runs.json")
DEFAULT_MODEL = "openai/gpt-5.2"
PROMPT_VERSION = "v2"
SEED = 42
MAX_ATTEMPTS = 3                      # bounded retries for transient failures only
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

SCHEMA = {
    "name": "album_tracklist",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "artist": {"type": "string"},
            "album_name": {"type": "string"},
            "release_year": {"type": "integer"},
            "tracks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"position": {"type": "integer"},
                                   "title": {"type": "string"}},
                    "required": ["position", "title"],
                    "additionalProperties": False,
                },
            },
            "excluded_candidates": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["artist", "album_name", "release_year", "tracks",
                     "excluded_candidates", "notes"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are a precise music-discography assistant. Answer only with the requested JSON. "
    "Never invent songs: if you are unsure about a detail, say so in `notes`."
)

USER_TEMPLATE = """YouTube video metadata:
- Title: {title}
- Channel: {channel}
- Description (first 400 characters): {description}

Tasks:
1. Identify the primary performing artist of this video (use the metadata ONLY for this).
2. Determine that artist's FIRST STUDIO ALBUM from your own knowledge of the discography
   (not necessarily the album mentioned in the description; exclude singles, EPs,
   compilations and live albums).
3. List the tracks of the ORIGINAL first-release standard edition (the original LP/CD as
   first issued), in original release order (side A then side B for vinyl).
   Include ONLY tracks you are confident were on that original edition. Exclude bonus
   tracks, tracks added by later reissues or regional editions, alternate versions and
   remixes. If you are not sure a track was on the original edition, EXCLUDE it.
4. Put every track you considered but excluded (bonus, reissue-only, uncertain) into
   `excluded_candidates`, and mention regional differences in `notes`.
Return `position` starting at 1 with no gaps."""


def load_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("[ERROR] OPENROUTER_API_KEY is not set (environment or .env).")
    return key


def build_messages(meta: dict) -> list:
    description = (meta.get("text") or meta.get("description") or "")[:400].replace("\n", " ")
    user = USER_TEMPLATE.format(title=meta.get("title", ""), channel=meta.get("channelName", ""),
                                description=description)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def call_openrouter(model: str, messages: list, key: str, with_sampling: bool = True,
                    max_tokens: int = 2500):
    """Return (parsed_json, raw_response, error). Never prints the key."""
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        "max_tokens": max_tokens,
    }
    if with_sampling:
        payload.update({"temperature": 0, "seed": SEED})
    resp, last_error = None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(API_URL, json=payload, timeout=180,
                                 headers={"Authorization": f"Bearer {key}"})
            last_error = None
        except requests.RequestException as exc:
            resp, last_error = None, f"network error: {exc.__class__.__name__}: {exc}"
        retryable = last_error is not None or resp.status_code in RETRY_STATUS
        if not retryable or attempt == MAX_ATTEMPTS:
            break
        wait = 2 * attempt
        reason = last_error or f"HTTP {resp.status_code}"
        print(f"  transient failure ({reason}); retry {attempt}/{MAX_ATTEMPTS - 1} in {wait}s",
              file=sys.stderr)
        time.sleep(wait)
    if last_error:
        return None, None, f"{last_error} (after {MAX_ATTEMPTS} attempts)"
    if resp.status_code != 200:
        hint = {401: "invalid OPENROUTER_API_KEY", 402: "insufficient OpenRouter credits"}.get(
            resp.status_code, "")
        return None, None, f"HTTP {resp.status_code} {hint}: {resp.text[:400]}".strip()
    try:
        raw = resp.json()
        choice = raw["choices"][0]
        if choice.get("finish_reason") == "length":
            return None, raw, ("output truncated (finish_reason=length): reasoning tokens "
                               "probably used up max_tokens; raise --max-tokens")
        return json.loads(choice["message"]["content"]), raw, None
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return None, None, f"unparseable response ({exc.__class__.__name__}): {resp.text[:400]}"


def validate(result: dict) -> list:
    problems = []
    tracks = result.get("tracks") or []
    if not result.get("artist", "").strip():
        problems.append("empty artist")
    if not result.get("album_name", "").strip():
        problems.append("empty album_name")
    if not tracks:
        problems.append("empty tracklist")
    if [t.get("position") for t in tracks] != list(range(1, len(tracks) + 1)):
        problems.append("positions are not 1..N in order")
    titles = [t.get("title", "").strip().lower() for t in tracks]
    if any(not t for t in titles):
        problems.append("empty track title")
    if len(set(titles)) != len(titles):
        problems.append("duplicate track titles")
    excluded = {t.strip().lower() for t in result.get("excluded_candidates", [])}
    if excluded & set(titles):
        problems.append("a track is both included and excluded")
    return problems


def signature(result: dict):
    return (result["artist"], result["album_name"], result["release_year"],
            tuple(t["title"] for t in result["tracks"]))


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2500,
                        help="Upper bound reserved on your OpenRouter credits (default 2500).")
    args = parser.parse_args()
    key = load_key()

    if not os.path.exists(YOUTUBE_PATH):
        print(f"[ERROR] {YOUTUBE_PATH} not found. Run spike_youtube.py first.", file=sys.stderr)
        return 1
    with open(YOUTUBE_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    messages = build_messages(meta)
    print(f"model: {args.model}   prompt: {PROMPT_VERSION}   runs: {args.runs}\n")

    runs, with_sampling = [], True
    for i in range(1, args.runs + 1):
        result, raw, error = call_openrouter(args.model, messages, key, with_sampling,
                                             args.max_tokens)
        if error and with_sampling and error.startswith("HTTP 400"):
            print(f"[run {i}] HTTP 400 with temperature/seed; retrying once without them.",
                  file=sys.stderr)
            with_sampling = False
            result, raw, error = call_openrouter(args.model, messages, key, with_sampling,
                                                 args.max_tokens)
        if error:
            print(f"[run {i}] [ERROR] {error}", file=sys.stderr)
            return 1
        problems = validate(result)
        usage = (raw or {}).get("usage", {})
        print(f"== run {i} ==  served_by_model: {raw.get('model')}  "
              f"temperature/seed sent: {with_sampling}  "
              f"finish: {raw['choices'][0].get('finish_reason')}  "
              f"tokens in/out: {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")
        print(f"  artist: {result['artist']}   album: {result['album_name']} "
              f"({result['release_year']})   tracks: {len(result['tracks'])}")
        print(f"  tracklist: {[t['title'] for t in result['tracks']]}")
        print(f"  excluded_candidates: {result['excluded_candidates']}")
        print(f"  notes: {result['notes']}")
        print(f"  validation problems: {problems or 'none'}\n")
        runs.append({"result": result, "problems": problems, "served_by": raw.get("model"),
                     "usage": usage, "sampling_params_sent": with_sampling})

    sigs = {signature(r["result"]) for r in runs}
    print(f"IDENTICAL across all {len(runs)} runs: {len(sigs) == 1}")
    if len(sigs) > 1:
        print("  -> the answers differ; this is the nondeterminism we must manage.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "prompt_version": PROMPT_VERSION, "runs": runs},
                  fh, ensure_ascii=False, indent=2)
    print(f"Saved to {OUT_PATH} (git-ignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())