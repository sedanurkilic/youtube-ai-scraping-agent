#!/usr/bin/env python3
"""
Development helper (NOT part of the submission): fill the local album cache from an
EARLIER real gpt-5.2 / prompt-v2 answer saved by spike_llm.py, so pipeline debugging does
not spend OpenRouter credits again.

    python3 seed_album_cache.py                      # uses spike_output/madonna/llm_runs.json
    python3 seed_album_cache.py --runs-file spike_output/llm_runs.json --video-id rSaC-YbSDpo

It refuses to seed from a different prompt version or model, and runs the same validation
as the pipeline. The cache lives in .cache/ (git-ignored); --no-cache bypasses it.
"""

import argparse
import json
import sys

import Seda_Nur_Kilic_YouTube_AI as core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-file", default="spike_output/madonna/llm_runs.json")
    parser.add_argument("--video-id", default="rSaC-YbSDpo")
    args = parser.parse_args()
    try:
        with open(args.runs_file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] cannot read {args.runs_file}: {exc}", file=sys.stderr)
        return 1
    if data.get("prompt_version") != core.PROMPT_VERSION or data.get("model") != core.DEFAULT_MODEL:
        print(f"[ERROR] {args.runs_file} was made with model={data.get('model')!r} "
              f"prompt={data.get('prompt_version')!r}; the pipeline uses "
              f"{core.DEFAULT_MODEL!r} / {core.PROMPT_VERSION!r}. Not seeding.", file=sys.stderr)
        return 1
    try:
        artist, album, tracks = core.validate_album(data["runs"][0]["result"])
    except (KeyError, IndexError, core.PipelineError) as exc:
        print(f"[ERROR] saved answer is not valid: {exc}", file=sys.stderr)
        return 1
    path = core._album_cache_path(args.video_id, core.DEFAULT_MODEL)
    core._write_json(path, {"artist": artist, "album": album, "tracks": tracks})
    print(f"seeded {path.name}: {artist} - {album}: {tracks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
