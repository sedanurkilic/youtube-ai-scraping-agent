#!/usr/bin/env python3
"""
Phase 2 / micro-step 2.1 - Genius reachability spike (development tool, NOT part of
the submission).

Question: can the permitted actor apify/web-scraper fetch ONE Genius lyrics page and
return the lyrics containers' text?

* Uses only Apify (APIFY_API_KEY from the environment or a local .env file).
* At most 2 attempts: (1) Apify default proxy, (2) residential proxy. The second
  attempt happens ONLY when the first run reached Genius but got no usable lyrics.
  Apify-side 4xx errors (permissions, auth, credits) stop immediately: a proxy
  change cannot fix them.
* Saves the RAW result to spike_output/genius_raw.json (git-ignored: copyrighted).
* Prints only numbers/diagnostics, never lyric text.
* No cleaning policy is applied here; that is step 2.2, done offline on the saved file.

Usage:
    export APIFY_API_KEY=...        # or put it in .env
    python3 spike_genius.py
    python3 spike_genius.py --url https://genius.com/Madonna-holiday-lyrics
"""

import argparse
import json
import os
import sys

import requests

ACTOR = "apify~web-scraper"
RUN_SYNC_URL = f"https://api.apify.com/v2/acts/{ACTOR}/run-sync-get-dataset-items"
DEFAULT_URL = "https://genius.com/Madonna-holiday-lyrics"
OUT_PATH = os.path.join("spike_output", "genius_raw.json")

PAGE_FUNCTION = """
async function pageFunction(context) {
    const { request, response } = context;
    const containers = Array.from(document.querySelectorAll('[data-lyrics-container="true"]'));
    return {
        url: request.url,
        httpStatus: response ? response.status : null,
        title: document.title,
        containerCount: containers.length,
        innerTexts: containers.map(c => c.innerText),
        textContents: containers.map(c => c.textContent),
        bodyHead: document.body ? document.body.innerText.slice(0, 300) : null
    };
}
"""

ATTEMPTS = [
    ("apify_default_proxy", {"useApifyProxy": True}),
    ("apify_residential_proxy", {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}),
]

HINTS = {
    401: "invalid APIFY_API_KEY",
    402: "plan/credits do not allow this run",
    408: "run timed out",
}


def load_token() -> str:
    try:
        from dotenv import load_dotenv  # optional convenience
        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("APIFY_API_KEY", "").strip()
    if not token:
        raise SystemExit("[ERROR] APIFY_API_KEY is not set (environment or .env).")
    return token


def run_actor(url: str, proxy: dict, token: str):
    """Return (items, error_message, fatal). Never prints the token.

    fatal=True means an Apify-side 4xx problem (auth, permissions, credits) that a
    different proxy cannot fix, so the caller must not retry.
    """
    payload = {
        "startUrls": [{"url": url}],
        "pageFunction": PAGE_FUNCTION,
        "proxyConfiguration": proxy,
        "runMode": "PRODUCTION",
        "maxPagesPerCrawl": 1,
        "maxRequestRetries": 1,
    }
    try:
        resp = requests.post(RUN_SYNC_URL, params={"timeout": 120}, json=payload,
                             headers={"Authorization": f"Bearer {token}"}, timeout=200)
    except requests.RequestException as exc:
        return None, f"network error: {exc.__class__.__name__}: {exc}", False

    if resp.status_code in (200, 201):
        try:
            return resp.json(), None, False
        except ValueError:
            return None, "Apify returned a non-JSON body.", False

    try:
        err = resp.json()["error"]
        detail = f"{err.get('type', '')}: {err.get('message', '')}".strip()
    except (ValueError, KeyError, TypeError, AttributeError):
        detail = resp.text[:500].replace("\n", " ")
    hint = HINTS.get(resp.status_code)
    message = f"HTTP {resp.status_code}" + (f" ({hint})" if hint else "") + f": {detail}"
    fatal = 400 <= resp.status_code < 500 and resp.status_code != 408
    return None, message, fatal


def summarize(item: dict) -> dict:
    joined = "\n".join(item.get("innerTexts") or [])
    lines = joined.split("\n")
    return {
        "http_status": item.get("httpStatus"),
        "page_title": item.get("title"),
        "container_count": item.get("containerCount"),
        "innertext_chars": len(joined),
        "line_count": len(lines),
        "empty_lines": sum(1 for ln in lines if not ln.strip()),
        "section_tag_lines": sum(1 for ln in lines if ln.strip().startswith("[")),
    }


def is_success(s: dict) -> bool:
    return s["http_status"] == 200 and (s["container_count"] or 0) > 0 and s["innertext_chars"] > 100


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    token = load_token()

    print(f"Target URL (a CANDIDATE, verify the title below): {args.url}")
    last_item, last_label, succeeded = None, None, False
    for label, proxy in ATTEMPTS:
        print(f"\n== Attempt: {label} ==")
        items, error, fatal = run_actor(args.url, proxy, token)
        if error:
            print(f"[ERROR] {error}", file=sys.stderr)
            if fatal:
                print("Apify-side error: a different proxy cannot fix this. Stopping "
                      "without further attempts.", file=sys.stderr)
                break
            continue
        if not items:
            print("[ERROR] Actor finished but returned no items.", file=sys.stderr)
            continue
        item = items[0]
        last_item, last_label = item, label
        summary = summarize(item)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if is_success(summary):
            print("  RESULT: lyrics containers found.")
            succeeded = True
            break
        print("  RESULT: no usable lyrics (blocked or layout changed).")
        if not summary["container_count"]:
            print(f"  body head (diagnostic): {item.get('bodyHead')!r}")

    if not succeeded:
        print("\nNo attempt produced usable lyrics. Report this block precisely; "
              "do not claim success.", file=sys.stderr)

    if last_item is not None:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump({"attempt": last_label, "item": last_item}, fh, ensure_ascii=False, indent=2)
        print(f"\nRaw item saved to {OUT_PATH} (git-ignored).")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
