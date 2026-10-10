#!/usr/bin/env python3
"""
Phase 2 / micro-step 2.2 - OFFLINE lyrics-policy calibration (development tool, NOT
part of the submission). Uses spike_output/genius_raw.json; no Apify credits, no
network. Prints NUMBERS ONLY - never lyric text.

Pre-registered candidates (7, fixed before looking at results):
  source innerText  : P1 as-is | P2 strip ends | P3 drop [section] lines (+strip)
                      | P4 P3 + drop blank lines | P5 P4 + strip each line
  source textContent: P6 as-is | P7 strip ends
Containers are always joined with "\\n".
Word rules compared: whitespace split() vs Unicode-aware regex.

Selection rule (decided in advance, unchanged):
  1. If one or more of P1-P7 reproduce the reference lyrics MD5 -> the first.
  2. Otherwise -> P2 (innerText, trim only the outer whitespace; nothing inside the
     lyrics is removed or rewritten) and the mismatch is documented.

DIAGNOSTIC rows D1..Dk (added after the first run showed a ~55% content gap):
  prefix of the first k NON-EMPTY lyrics containers. They are printed for
  understanding only and are EXCLUDED from selection. A diagnostic counts as
  evidence only if chars, words, tokens AND md5 all match at once; a single matching
  number is treated as coincidence.
"""

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Seda_Nur_Kilic_YouTube_AI as core  # noqa: E402

RAW_PATH = os.path.join("spike_output", "genius_raw.json")
LOG_PATH = os.path.join("spike_output", "lyrics_calibration_log.json")
REF = {"chars": 1482, "words": 264, "tokens": 395,
       "md5": "1933a103bbc2c282de279e859608e189"}

TAG_LINE = re.compile(r"^\s*\[[^\]]*\]\s*$")
WORD_RE = re.compile(r"\w+(?:['\u2019]\w+)*", re.UNICODE)


def words_split(text):
    return len(text.split())


def words_regex(text):
    return len(WORD_RE.findall(text))


def drop_tags(text):
    return "\n".join(ln for ln in text.split("\n") if not TAG_LINE.match(ln))


def drop_blank(text):
    return "\n".join(ln for ln in text.split("\n") if ln.strip())


def policies(inner, content):
    p3 = drop_tags(inner).strip()
    p4 = drop_blank(p3)
    return [
        ("P1 innerText as-is", inner),
        ("P2 innerText strip", inner.strip()),
        ("P3 no [tags]", p3),
        ("P4 no tags/blank", p4),
        ("P5 P4+strip lines", "\n".join(ln.strip() for ln in p4.split("\n"))),
        ("P6 textContent as-is", content),
        ("P7 textContent strip", content.strip()),
    ]


def diagnostics(inner_parts):
    """Prefixes of the non-empty containers. Diagnostic only, never selected."""
    nonempty = [p for p in inner_parts if p.strip()]
    return [(f"D{k} first {k} non-empty", "\n".join(nonempty[:k]).strip())
            for k in range(1, len(nonempty) + 1)]


def evaluate(name, text):
    tokens = core.count_tokens(text)
    ws, wr = words_split(text), words_regex(text)
    md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
    return {
        "policy": name, "chars": len(text), "words_split": ws, "words_regex": wr,
        "tokens": tokens, "md5": md5,
        "hit_chars": len(text) == REF["chars"],
        "hit_words": REF["words"] in (ws, wr),
        "hit_tokens": tokens == REF["tokens"],
        "hit_md5": md5 == REF["md5"],
    }


def print_table(rows):
    head = f"{'policy':<24}{'chars':>6}{'w_split':>9}{'w_regex':>9}{'tokens':>8}  {'md5':<10}hits"
    print(head)
    print("-" * len(head))
    for r in rows:
        hits = "".join(k for k, ok in (("C", r["hit_chars"]), ("W", r["hit_words"]),
                                       ("T", r["hit_tokens"]), ("M", r["hit_md5"])) if ok)
        print(f"{r['policy']:<24}{r['chars']:>6}{r['words_split']:>9}{r['words_regex']:>9}"
              f"{r['tokens']:>8}  {r['md5'][:8]:<10}{hits or '-'}")


def main() -> int:
    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] {RAW_PATH} not found. Run spike_genius.py first.", file=sys.stderr)
        return 1
    with open(RAW_PATH, encoding="utf-8") as fh:
        item = json.load(fh)["item"]
    inner_parts = item.get("innerTexts") or []
    content_parts = item.get("textContents") or []
    if not inner_parts:
        print("[ERROR] Raw file has no innerTexts.", file=sys.stderr)
        return 1

    print(f"page_title: {item.get('title')}")
    print(f"containers: {len(inner_parts)}  innerText chars per container: "
          f"{[len(p) for p in inner_parts]}")
    print(f"reference targets: chars={REF['chars']} words={REF['words']} "
          f"tokens={REF['tokens']} md5={REF['md5'][:8]}...\n")

    print("== Pre-registered policies (selection candidates) ==")
    rows = [evaluate(n, t) for n, t in
            policies("\n".join(inner_parts), "\n".join(content_parts))]
    print_table(rows)

    matched = [r for r in rows if r["hit_md5"]]
    selected = matched[0] if matched else rows[1]
    reason = "md5_match" if matched else "default_P2_reference_not_reproduced"
    print(f"\nSELECTED: {selected['policy']} -> {reason}")
    if not matched:
        print("NOTE: no policy reproduced the reference MD5. The reference is a "
              "target, not a fixture; document the gap, do not invent a policy to hit it.")

    print("\n== Diagnostics (NOT selection candidates) ==")
    diag = [evaluate(n, t) for n, t in diagnostics(inner_parts)]
    print_table(diag)
    full = [d["policy"] for d in diag
            if d["hit_chars"] and d["hit_words"] and d["hit_tokens"] and d["hit_md5"]]
    print("\nAll four numbers match at once: "
          f"{', '.join(full) if full else 'none (no diagnostic is evidence)'}")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump({"reference": REF, "rows": rows, "diagnostics": diag,
                   "selected": selected["policy"], "reason": reason}, fh, indent=2)
    print(f"Numbers-only log written to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
