#!/usr/bin/env python3
"""
Phase 3 / micro-step 3.2 - golden-set evaluation of the album-discovery step
(development tool, NOT part of the submission).

For every case in tests/golden_albums.json:
  1. get the YouTube metadata (Apify, cached in spike_output/cases/<video_id>/);
  2. GUARD: the video title must contain `expect_title_contains`; otherwise the case
     fails as "wrong video" and no LLM credits are spent;
  3. ask the LLM N times with the production prompt (spike_llm.py);
  4. compare artist, album and tracklist with the independently verified truth and
     check that all runs agree.

Tracklist rule (compare_tracklist): after normalising titles, the answer must be
exactly `required` in order, optionally followed by items from `optional`; any other
title (bonus/remix/reissue track), any missing/duplicated/reordered track fails.

Usage:
    python3 eval_golden.py                 # all cases, 2 runs each
    python3 eval_golden.py --only adele_rolling_in_the_deep --runs 3
"""

import argparse
import json
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Seda_Nur_Kilic_YouTube_AI as core  # noqa: E402
import spike_llm as llm  # noqa: E402
import spike_youtube as yt  # noqa: E402

class OutOfCredits(RuntimeError):
    """OpenRouter refused with HTTP 402: every further call would fail too."""


GOLDEN_PATH = os.path.join("tests", "golden_albums.json")
CASES_DIR = os.path.join("spike_output", "cases")


# ------------------------------ pure comparison logic -------------------------------
def norm(text: str) -> str:
    """Case/punctuation-insensitive form used ONLY for comparing titles in tests."""
    text = text.lower().replace("\u2019", "'").replace("'", "")  # don't -> dont
    text = re.sub(r"[^\w\s]", " ", text)                         # (No,No) -> no no
    return re.sub(r"\s+", " ", text).strip()


def compare_tracklist(required, optional, got) -> dict:
    req, opt = [norm(t) for t in required], [norm(t) for t in optional]
    g = [norm(t) for t in got]
    req_set, opt_set, g_set = set(req), set(opt), set(g)

    extra = [t for t in got if norm(t) not in req_set | opt_set]
    missing = [t for t in required if norm(t) not in g_set]
    optional_used = [t for t in got if norm(t) in opt_set]
    duplicates = len(g) != len(g_set)

    core_in_answer = [x for x in g if x in req_set]
    core_expected = [x for x in req if x in g_set]
    order_ok = core_in_answer == core_expected

    last_req = max((i for i, x in enumerate(g) if x in req_set), default=-1)
    first_opt = min((i for i, x in enumerate(g) if x in opt_set), default=len(g))
    optional_after_required = first_opt > last_req

    ok = not (extra or missing or duplicates) and order_ok and optional_after_required
    return {"ok": ok, "extra": extra, "missing": missing, "optional_used": optional_used,
            "duplicates": duplicates, "order_ok": order_ok,
            "optional_after_required": optional_after_required}


def describe_comparison(cmp: dict) -> str:
    if cmp["ok"]:
        used = f" (+ optional: {cmp['optional_used']})" if cmp["optional_used"] else ""
        return "tracklist matches" + used
    parts = []
    if cmp["extra"]:
        parts.append(f"unexpected tracks: {cmp['extra']}")
    if cmp["missing"]:
        parts.append(f"missing tracks: {cmp['missing']}")
    if cmp["duplicates"]:
        parts.append("duplicate tracks")
    if not cmp["order_ok"]:
        parts.append("required tracks are in the wrong order")
    if not cmp["optional_after_required"]:
        parts.append("an optional track appears before the end of the required ones")
    return "; ".join(parts)


# ------------------------------ I/O helpers -----------------------------------------
def video_id(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    if "v" in query:
        return query["v"][0]
    return urlparse(url).path.rstrip("/").split("/")[-1]


def get_metadata(case: dict, token: str):
    """Return (meta, source, error). Cached so Apify is called once per video."""
    folder = os.path.join(CASES_DIR, video_id(case["url"]))
    path = os.path.join(folder, "youtube.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), "cache", None
    items, error = yt.run_actor(case["url"], token)
    if error:
        return None, "apify", error
    if not items:
        return None, "apify", "no items returned (private/unavailable video?)"
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items[0], fh, ensure_ascii=False, indent=2)
    return items[0], "apify", None


def ask_llm(model, messages, key, max_tokens):
    result, _raw, error = llm.call_openrouter(model, messages, key, True, max_tokens)
    if error and error.startswith("HTTP 400"):  # model rejected temperature/seed
        result, _raw, error = llm.call_openrouter(model, messages, key, False, max_tokens)
    return result, error


def cleaned_signature(result: dict):
    """Run-to-run identity as the pipeline sees it: exact strings, titles cleaned."""
    return (result["artist"], result["album_name"], result["release_year"],
            tuple(core.clean_track_title(t["title"]) for t in result["tracks"]))


def evaluate_run(case: dict, result: dict) -> dict:
    problems = llm.validate(result)
    cmp = compare_tracklist(case["required"], case.get("optional", []),
                            [core.clean_track_title(t["title"]) for t in result["tracks"]])
    artist_ok = norm(result["artist"]) == norm(case["artist"])
    album_ok = norm(result["album_name"]) == norm(case["album"])
    return {"ok": cmp["ok"] and artist_ok and album_ok and not problems,
            "artist_ok": artist_ok, "album_ok": album_ok, "problems": problems,
            "comparison": cmp}


def run_case(case, runs, model, max_tokens, yt_token, or_key) -> str:
    print(f"\n=== {case['id']} ===")
    meta, source, error = get_metadata(case, yt_token)
    if error:
        print(f"  ERROR (YouTube metadata): {error}")
        return "ERROR"
    title = meta.get("title", "")
    if case["expect_title_contains"].lower() not in title.lower():
        print(f"  FAIL - wrong video: title {title!r} does not contain "
              f"{case['expect_title_contains']!r}. Fix the URL in the golden file.")
        return "FAIL"
    print(f"  video: {title!r} (metadata from {source})")

    messages = llm.build_messages(meta)
    results, verdicts = [], []
    for i in range(1, runs + 1):
        result, error = ask_llm(model, messages, or_key, max_tokens)
        if error:
            if error.startswith("HTTP 402"):
                raise OutOfCredits(error)
            print(f"  run {i}: ERROR (LLM): {error[:250]}")
            return "ERROR"
        verdict = evaluate_run(case, result)
        results.append(result)
        verdicts.append(verdict)
        status = "PASS" if verdict["ok"] else "FAIL"
        print(f"  run {i}: {status} | artist={result['artist']!r} album={result['album_name']!r} "
              f"tracks={len(result['tracks'])}")
        print(f"         {describe_comparison(verdict['comparison'])}")
        if not verdict["artist_ok"]:
            print(f"         artist differs from expected {case['artist']!r}")
        if not verdict["album_ok"]:
            print(f"         album differs from expected {case['album']!r}")
        if verdict["problems"]:
            print(f"         validation problems: {verdict['problems']}")

    consistent = len({cleaned_signature(r) for r in results}) == 1
    print(f"  consistent across {runs} runs: {consistent}")
    passed = all(v["ok"] for v in verdicts) and consistent
    print(f"  => {'PASS' if passed else 'FAIL'}")
    return "PASS" if passed else "FAIL"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--model", default=llm.DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--only", default=None, help="Run a single case id.")
    args = parser.parse_args()

    with open(args.golden, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            print(f"[ERROR] no case with id {args.only!r}", file=sys.stderr)
            return 1

    yt_token, or_key = yt.load_token(), llm.load_key()
    print(f"model: {args.model}   prompt: {llm.PROMPT_VERSION}   runs per case: {args.runs}")
    outcomes = {}
    for c in cases:
        try:
            outcomes[c["id"]] = run_case(c, args.runs, args.model, args.max_tokens,
                                         yt_token, or_key)
        except OutOfCredits:
            print("\n[STOP] OpenRouter credits are insufficient for this request "
                  f"(--max-tokens {args.max_tokens}). Add credits or lower --max-tokens; "
                  "the remaining cases were not run.")
            outcomes[c["id"]] = "ERROR"
            break

    print("\n== Summary ==")
    for case_id, outcome in outcomes.items():
        print(f"  {outcome:<6}{case_id}")
    all_pass = all(o == "PASS" for o in outcomes.values())
    print("ALL PASS" if all_pass else "NOT ALL PASSING - do not freeze the prompt yet.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())