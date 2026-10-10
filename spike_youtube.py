#!/usr/bin/env python3
"""
Phase 2 / micro-step 2.3 - YouTube metadata spike (development tool, NOT part of the
submission).

Question: what does the permitted actor streamers/youtube-scraper return for the
assessment's YouTube URL, and which fields identify the artist?

* Only Apify; APIFY_API_KEY from the environment or .env.
* One run. Saves the RAW item to spike_output/youtube_raw.json (git-ignored).
* Prints the field NAMES the actor returns plus a few identifying fields
  (title, channel, date, a 300-character description snippet).

Usage:
    python3 spike_youtube.py
    python3 spike_youtube.py --url "https://www.youtube.com/watch?v=rSaC-YbSDpo"
"""

import argparse
import json
import os
import sys

import requests

ACTOR = "streamers~youtube-scraper"
RUN_SYNC_URL = f"https://api.apify.com/v2/acts/{ACTOR}/run-sync-get-dataset-items"
DEFAULT_URL = "https://www.youtube.com/watch?v=rSaC-YbSDpo"
OUT_PATH = os.path.join("spike_output", "youtube_raw.json")

HINTS = {401: "invalid APIFY_API_KEY", 402: "plan/credits do not allow this run",
         408: "run timed out"}
SHOW_FIELDS = ("title", "channelName", "channelUrl", "id", "date", "duration", "viewCount")
DESCRIPTION_FIELDS = ("text", "description")


def load_token() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("APIFY_API_KEY", "").strip()
    if not token:
        raise SystemExit("[ERROR] APIFY_API_KEY is not set (environment or .env).")
    return token


def run_actor(url: str, token: str):
    """Return (items, error_message). Never prints the token."""
    payload = {"startUrls": [{"url": url}]}
    try:
        resp = requests.post(RUN_SYNC_URL, params={"timeout": 120}, json=payload,
                             headers={"Authorization": f"Bearer {token}"}, timeout=200)
    except requests.RequestException as exc:
        return None, f"network error: {exc.__class__.__name__}: {exc}"
    if resp.status_code in (200, 201):
        try:
            return resp.json(), None
        except ValueError:
            return None, "Apify returned a non-JSON body."
    try:
        err = resp.json()["error"]
        detail = f"{err.get('type', '')}: {err.get('message', '')}".strip()
        if err.get("approvalUrl"):
            detail += f" | Approve here: {err['approvalUrl']}"
    except (ValueError, KeyError, TypeError, AttributeError):
        detail = resp.text[:500].replace("\n", " ")
    hint = HINTS.get(resp.status_code)
    return None, f"HTTP {resp.status_code}" + (f" ({hint})" if hint else "") + f": {detail}"


def describe(item: dict) -> None:
    print(f"fields returned ({len(item)}): {sorted(item.keys())}\n")
    for key in SHOW_FIELDS:
        if key in item:
            print(f"  {key}: {item[key]!r}")
    for key in DESCRIPTION_FIELDS:
        if isinstance(item.get(key), str) and item[key]:
            snippet = item[key][:300].replace("\n", " | ")
            print(f"  {key} (first 300 chars): {snippet!r}")
            break


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    token = load_token()

    print(f"Target URL: {args.url}\n")
    items, error = run_actor(args.url, token)
    if error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    if not items:
        print("[ERROR] Actor finished but returned no items (private/unavailable video?).",
              file=sys.stderr)
        return 1
    if len(items) > 1:
        print(f"note: {len(items)} items returned; using the first.\n")
    item = items[0]
    describe(item)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(item, fh, ensure_ascii=False, indent=2)
    print(f"\nRaw item saved to {OUT_PATH} (git-ignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
