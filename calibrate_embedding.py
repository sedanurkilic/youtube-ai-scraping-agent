#!/usr/bin/env python3
"""
Phase 1 / micro-step 1.1 - OFFLINE embedding calibration (no Apify, no OpenRouter).

THIS IS A DEVELOPMENT TOOL, NOT PART OF THE SUBMISSION. The final single-file
CLI will contain only the ONE chosen configuration; it never runs this matrix.

Question this script answers:
    Does any documented way of calling nomic-ai/nomic-embed-text-v1.5 on the exact
    string "395,703,470,622,561,950,712,634" reproduce the reference MD5?

Rules (from the roadmap):
  * At most SIX pre-registered trials: 2 prefixes x 3 post-processings.
  * Loader, device (CPU), dtype (float32) and revision are FIXED for the matrix.
  * The reference MD5 is only COMPARED against; it never tunes anything.
  * Every trial is logged (console + calibration_log.json).

Selection rule (decided BEFORE seeing results):
  1. If one or more trials match the reference MD5 -> select the first match in the
     fixed trial order (= least post-processing).
  2. Otherwise -> select T1 (prefix none + native output). It is the closest to the
     literal assessment text (input string -> embedding -> format -> MD5) and
     modifies nothing. It is reported as UNVERIFIED against the reference, because
     a mismatch cannot be blamed on the Nomic configuration alone (the reference
     may come from a different/untruncated source pipeline).

Usage:
    python calibrate_embedding.py                    # resolves latest SHA, then pins it
    python calibrate_embedding.py --revision <sha>   # use an explicit commit
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone

# ----------------------------- fixed experiment inputs -----------------------------
INPUT_TEXT = "395,703,470,622,561,950,712,634"
REFERENCE_MD5 = "4ae6acfbd3ccc56d4a62b3f3a4f3eb1f"  # compared against only
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
EXPECTED_DIM = 768
EXPECTED_DTYPE = "torch.float32"

# Pre-registered matrix (2 x 3 = 6 trials). Order is fixed: T1..T6.
PREFIXES = {
    "none": "",
    "search_document": "search_document: ",
}
PROCESSINGS = ["native", "l2", "layernorm_l2"]


# ----------------------------- pure, reusable functions ----------------------------
def serialize_embedding(vector) -> str:
    """Assessment rule: each float -> exactly 10 decimals, comma-joined, no spaces."""
    return ",".join(f"{float(x):.10f}" for x in vector)


def md5_hex(text: str) -> str:
    """MD5 of the UTF-8 bytes, lowercase hex (32 chars)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def apply_processing(raw, mode: str):
    """
    raw: 2-D torch tensor (1, 768) straight from model.encode().
    native       -> unchanged (whatever the loader returns)
    l2           -> L2 normalization only
    layernorm_l2 -> official Matryoshka recipe WITHOUT truncation:
                    layer_norm -> (keep all 768 dims) -> L2 normalization
    """
    import torch.nn.functional as F

    if mode == "native":
        return raw
    if mode == "l2":
        return F.normalize(raw, p=2, dim=1)
    if mode == "layernorm_l2":
        x = F.layer_norm(raw, normalized_shape=(raw.shape[1],))
        return F.normalize(x, p=2, dim=1)
    raise ValueError(f"unknown processing mode: {mode}")


def select_configuration(trials: list):
    """Pre-declared selection rule. Returns (trial, reason)."""
    matched = [t for t in trials if t["matches_reference"]]
    if matched:
        return matched[0], "reference_match"
    literal = next(t for t in trials
                   if t["prefix"] == "none" and t["processing"] == "native")
    return literal, "literal_spec_fallback_UNVERIFIED_against_reference"


# ----------------------------- environment / model ---------------------------------
def collect_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "sentence_transformers", "transformers",
                 "numpy", "einops", "huggingface_hub"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            versions[name] = f"NOT INSTALLED ({exc.__class__.__name__})"
    return versions


def resolve_revision(explicit):
    """Return the exact commit SHA that WILL be loaded. Never returns 'unknown'."""
    if explicit:
        return explicit
    from huggingface_hub import HfApi
    sha = HfApi().model_info(MODEL_ID).sha
    if not sha:
        raise RuntimeError("Hugging Face returned no commit SHA for the model.")
    return sha


def load_model(revision: str):
    """Always loads the SAME revision that was logged."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, trust_remote_code=True,
                               device="cpu", revision=revision)


def encode_raw(model, text):
    """One encode call -> validated (1, 768) float32 CPU tensor, no normalization."""
    raw = model.encode([text], convert_to_tensor=True,
                       normalize_embeddings=False).detach().cpu()
    if str(raw.dtype) != EXPECTED_DTYPE:
        raise RuntimeError(f"Unexpected dtype {raw.dtype}; expected {EXPECTED_DTYPE}.")
    if tuple(raw.shape) != (1, EXPECTED_DIM):
        raise RuntimeError(f"Unexpected shape {tuple(raw.shape)}; expected (1, {EXPECTED_DIM}).")
    return raw


# ----------------------------- experiment ------------------------------------------
def run_matrix(model) -> list:
    import torch

    trials = []
    trial_no = 0
    for prefix_name, prefix in PREFIXES.items():
        text = prefix + INPUT_TEXT
        raw_a = encode_raw(model, text)
        raw_b = encode_raw(model, text)  # repeat -> determinism check
        repeatable = bool(torch.equal(raw_a, raw_b))
        max_abs_diff = float((raw_a - raw_b).abs().max().item())

        for mode in PROCESSINGS:
            trial_no += 1
            out = apply_processing(raw_a, mode)[0]
            vec = out.tolist()

            if not all(math.isfinite(v) for v in vec):
                raise ValueError(
                    f"Embedding contains NaN or Infinity "
                    f"(prefix={prefix_name}, processing={mode}); refusing to hash it."
                )

            serialized = serialize_embedding(vec)
            digest = md5_hex(serialized)

            trials.append({
                "trial_id": f"T{trial_no}",
                "prefix": prefix_name,
                "input_text": text,
                "input_md5": md5_hex(text),
                "processing": mode,
                "dtype": str(raw_a.dtype),
                "dim": len(vec),
                "dim_ok": len(vec) == EXPECTED_DIM,
                "finite_ok": all(math.isfinite(v) for v in vec),
                "l2_norm": round(float(out.norm()), 6),
                "repeat_identical": repeatable,
                "max_abs_diff_between_repeats": max_abs_diff,
                "md5": digest,
                "matches_reference": digest == REFERENCE_MD5,
            })
    return trials


def print_table(trials: list) -> None:
    header = (f"{'ID':<4}{'prefix':<17}{'processing':<14}{'dtype':<15}{'dim':<5}"
              f"{'|v|':<10}{'repeat':<8}{'maxdiff':<9}{'md5':<34}match")
    print(header)
    print("-" * len(header))
    for t in trials:
        print(f"{t['trial_id']:<4}{t['prefix']:<17}{t['processing']:<14}{t['dtype']:<15}"
              f"{t['dim']:<5}{t['l2_norm']:<10}{str(t['repeat_identical']):<8}"
              f"{t['max_abs_diff_between_repeats']:<9.1e}{t['md5']:<34}"
              f"{'YES' if t['matches_reference'] else 'no'}")


def fail(message: str) -> int:
    print(f"[ERROR] {message}", file=sys.stderr)
    print("Paste this error and the environment block; do NOT change versions blindly.",
          file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default=None,
                        help="Pin a Hugging Face commit SHA for the model repo.")
    parser.add_argument("--out", default="calibration_log.json")
    args = parser.parse_args()

    versions = collect_versions()
    print("== Environment ==")
    for k, v in versions.items():
        print(f"  {k}: {v}")

    try:
        revision = resolve_revision(args.revision)
    except Exception as exc:  # noqa: BLE001
        return fail(f"Could not determine model revision ({exc.__class__.__name__}: {exc}). "
                    "Refusing to run unpinned; pass --revision <sha>.")
    print(f"  model: {MODEL_ID}")
    print(f"  model revision (pinned and loaded): {revision}")
    print("  note: trust_remote_code also fetches modeling code from a second repo "
          "(nomic-bert-2048) that this script does not pin.")
    print(f"  input text: {INPUT_TEXT!r}\n")

    try:
        model = load_model(revision)
    except Exception as exc:  # noqa: BLE001
        return fail(f"Model load failed: {exc.__class__.__name__}: {exc}")

    try:
        trials = run_matrix(model)
    except Exception as exc:  # noqa: BLE001
        return fail(f"Embedding run failed: {exc.__class__.__name__}: {exc}")

    print("== Trials (max 6, pre-registered) ==")
    print_table(trials)

    selected, reason = select_configuration(trials)
    matched_ids = [t["trial_id"] for t in trials if t["matches_reference"]]
    print()
    print(f"Reference MD5 matched by: {', '.join(matched_ids) if matched_ids else 'none'}")
    print(f"SELECTED: {selected['trial_id']} "
          f"(prefix={selected['prefix']}, processing={selected['processing']}) -> {reason}")
    if not matched_ids:
        print("NOTE: mismatch is a documented open finding. It does not prove the Nomic "
              "configuration is wrong: the reference may come from a different source "
              "pipeline. Do not add trials; do not alter the input text.")

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_md5": REFERENCE_MD5,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "versions": versions,
        "selected_trial": selected["trial_id"],
        "selection_reason": reason,
        "trials": trials,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print(f"\nLog written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
