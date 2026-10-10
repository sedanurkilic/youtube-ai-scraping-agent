#!/usr/bin/env python3
"""
Seda_Nur_Kilic_YouTube_AI.py - YouTube AI Scraping Agent (WORKING COPY).

Usage:
    python3 Seda_Nur_Kilic_YouTube_AI.py "<youtube url>" json   # analysis JSON only on stdout
    python3 Seda_Nur_Kilic_YouTube_AI.py "<youtube url>" hash   # final 32-char MD5 only on stdout
    (optional flag: --no-cache)

Pipeline:
    YouTube URL -> Apify streamers/youtube-scraper (metadata)
      -> OpenRouter LLM (artist, first studio album, original tracklist; strict JSON)
      -> Genius lyrics via Apify apify/web-scraper (slug guess first, verified by page title;
         apify/google-search-scraper only as a fallback)
      -> per-song metrics (chars / words / cl100k_base tokens / tokens-per-word / MD5)
      -> JSON, or token-count string -> nomic-embed-text-v1.5 -> fixed 10-decimal string -> MD5

stdout carries ONLY the requested result; logs and errors go to stderr (exit code 1 on failure).
Keys: APIFY_API_KEY and OPENROUTER_API_KEY (environment or a local .env file).

Documented decisions (see README once written):
    * Characters  : len(lyrics) - Unicode code points, no normalization.
    * Words       : len(lyrics.split()) - PROVISIONAL; Phase 2 compares it with the
                    Holiday reference (264) and may replace it.
    * Tokens      : tiktoken cl100k_base, special-token strings treated as plain text.
    * Ratio/avg   : Python round(x, 2) (round-half-to-even on the exact binary value).
    * Zero words  : validation error (purported lyrics must contain at least one word).
    * Lyrics MD5  : MD5 of the UTF-8 bytes of the exact string used for all counts.
    * Lyrics text : innerText of Genius' lyrics containers joined with "\\n", outer whitespace
                    trimmed (calibration policy P2); nothing inside is removed or rewritten.
    * Embedding   : nomic-embed-text-v1.5, pinned revision, CPU, float32, no task
                    prefix, native output (calibration trial T1).
                    NOT verified against the assessment's reference hash.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

# ----------------------------------- constants -------------------------------------
ENCODING_NAME = "cl100k_base"
ROUND_DIGITS = 2

NOMIC_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
EMBEDDING_DIM = 768

# Public per-song fields, in the order of the assessment's example JSON.
SONG_METRIC_KEYS = (
    "lyrics_length_chars",
    "lyrics_length_words",
    "lyrics_length_tokens",
    "tokens_per_word",
    "lyrics_hash",
)


# ------------------------------- lyrics metrics ------------------------------------
@functools.lru_cache(maxsize=1)
def get_encoder():
    """Load the tiktoken encoder once (first use may download the BPE file)."""
    import tiktoken

    return tiktoken.get_encoding(ENCODING_NAME)


def count_chars(lyrics: str) -> int:
    """Unicode code points. No normalization: 'e\\u0301' is 2, 'é' is 1."""
    return len(lyrics)


def count_words(lyrics: str) -> int:
    """Whitespace-delimited words (PROVISIONAL rule, to be calibrated in Phase 2)."""
    return len(lyrics.split())


def count_tokens(lyrics: str) -> int:
    """cl100k_base token count. Strings like '<|endoftext|>' are plain text, not errors."""
    return len(get_encoder().encode(lyrics, disallowed_special=()))


def lyrics_md5(lyrics: str) -> str:
    """MD5 (lowercase hex) of the UTF-8 bytes of the exact lyrics string."""
    return hashlib.md5(lyrics.encode("utf-8")).hexdigest()


def tokens_per_word(tokens: int, words: int) -> float:
    if words <= 0:
        raise ValueError("Cannot compute tokens-per-word with zero words.")
    return round(tokens / words, ROUND_DIGITS)


def lyrics_metrics(lyrics: str) -> dict:
    """All per-song metrics from ONE preserved string. Never rewrites the lyrics."""
    if not isinstance(lyrics, str):
        raise TypeError(f"lyrics must be str, got {type(lyrics).__name__}")
    words = count_words(lyrics)
    if words == 0:
        raise ValueError("Lyrics contain zero words; refusing to treat them as valid lyrics.")
    tokens = count_tokens(lyrics)
    return {
        "lyrics_length_chars": count_chars(lyrics),
        "lyrics_length_words": words,
        "lyrics_length_tokens": tokens,
        "tokens_per_word": tokens_per_word(tokens, words),
        "lyrics_hash": lyrics_md5(lyrics),
    }


# ------------------------------- track titles --------------------------------------
_FEAT_IN_BRACKETS = re.compile(r"\s*[(\[]\s*(?:feat\.?|ft\.?|featuring)\s[^)\]]*[)\]]\s*$", re.IGNORECASE)
_FEAT_AFTER_DASH = re.compile(r"\s+-\s+(?:feat\.?|ft\.?|featuring)\s.*$", re.IGNORECASE)


def clean_track_title(title: str) -> str:
    """
    Remove a TRAILING featured-artist credit so the same song always gets the same name:
    'X (feat. Y)', 'X [ft. Y]', 'X (featuring Y)', 'X - feat. Y' -> 'X'.
    Everything else is left untouched ('(No, No, No)', '(Remix)', '(with Y)', ...).
    Never returns an empty string.
    """
    original = title.strip()
    cleaned = _FEAT_IN_BRACKETS.sub("", original)
    cleaned = _FEAT_AFTER_DASH.sub("", cleaned).strip()
    return cleaned or original


# ------------------------------- public JSON ---------------------------------------
def build_analysis(artist: str, album_name: str, tracks) -> dict:
    """
    tracks: iterable of {"track_index": int, "name": str, "metrics": lyrics_metrics(...)}
    Output order follows track_index (original release order), no matter in which
    order the tracks were completed. Internal fields never reach the public JSON.
    """
    ordered = sorted(tracks, key=lambda t: t["track_index"])
    if not ordered:
        raise ValueError("No tracks to analyse.")
    indices = [t["track_index"] for t in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate track_index values: {indices}")

    songs = [
        {"name": t["name"], **{k: t["metrics"][k] for k in SONG_METRIC_KEYS}}
        for t in ordered
    ]
    total_tokens = sum(s["lyrics_length_tokens"] for s in songs)
    return {
        "artist": artist,
        "album_name": album_name,
        "songs": songs,
        "total_tokens_all_songs": total_tokens,
        "avg_tokens_per_song": round(total_tokens / len(songs), ROUND_DIGITS),
    }


def token_count_string(analysis: dict) -> str:
    """'395,703,470,...' - token counts in album order, comma only, no spaces."""
    return ",".join(str(s["lyrics_length_tokens"]) for s in analysis["songs"])


# ------------------------------- embedding hash ------------------------------------
def serialize_embedding(vector) -> str:
    """Each float -> exactly 10 decimals (f-string), joined by ',' with no spaces."""
    return ",".join(f"{float(x):.10f}" for x in vector)


def embedding_hash_from_vector(vector) -> str:
    """Validate (768 finite numbers), serialize, MD5 of the UTF-8 text."""
    values = list(vector)
    if len(values) != EMBEDDING_DIM:
        raise ValueError(f"Embedding has {len(values)} values; expected {EMBEDDING_DIM}.")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Embedding contains NaN or Infinity.")
    return hashlib.md5(serialize_embedding(values).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=1)
def _load_nomic():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        NOMIC_MODEL_ID, trust_remote_code=True, device="cpu", revision=NOMIC_REVISION
    )


def embed_text(text: str) -> list:
    """Calibration trial T1: no prefix, native output, CPU, float32, pinned revision."""
    raw = (
        _load_nomic()
        .encode([text], convert_to_tensor=True, normalize_embeddings=False)
        .detach()
        .cpu()
    )
    if str(raw.dtype) != "torch.float32" or tuple(raw.shape) != (1, EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected embedding dtype/shape: {raw.dtype} {tuple(raw.shape)}")
    return raw[0].tolist()


def compute_embedding_hash(token_counts: str) -> str:
    """Final assessment hash for a token-count string such as '395,703,470'."""
    return embedding_hash_from_vector(embed_text(token_counts))


# =====================================================================================
# PIPELINE: YouTube -> LLM -> Genius -> metrics -> JSON / hash
# =====================================================================================
PIPELINE_VERSION = "1"
PROMPT_VERSION = "v2"
DEFAULT_MODEL = "openai/gpt-5.2"       # the model the assessment is graded with
DEFAULT_MAX_TOKENS = 3000              # override with OPENROUTER_MAX_TOKENS (e.g. low credit)
LLM_SEED = 42
MAX_ATTEMPTS = 3                       # bounded retries, transient failures only
RETRY_STATUS = {429, 500, 502, 503, 504}

APIFY_YOUTUBE = "streamers~youtube-scraper"
APIFY_WEB = "apify~web-scraper"
APIFY_SEARCH = "apify~google-search-scraper"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"   # metrics only, never lyrics

GENIUS_URL_RE = re.compile(r"^https?://genius\.com/[^/?#]+-lyrics/?$")
# A Genius page that is a different VERSION of the song is rejected unless the wanted
# track name itself contains that word.
VARIANT_WORDS = {"remix", "live", "demo", "instrumental", "acoustic", "translation",
                 "translations", "traduccion", "traducao", "ubersetzung", "turkce",
                 "romanized", "karaoke", "cover"}


class PipelineError(Exception):
    """A failure with the pipeline step that caused it, printed to stderr by main()."""

    def __init__(self, step: str, message: str):
        super().__init__(f"{step}: {message}")
        self.step = step


def log(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr, flush=True)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PipelineError("config", f"{name} is not set (environment variable or .env file).")
    return value


def _norm(text: str) -> str:
    """Accent/case/punctuation-insensitive form, used only to COMPARE titles."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------- input -----------------------------------------
def parse_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    video_id = None
    if host == "youtube.com" or host.endswith(".youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if not video_id:
            match = re.match(r"^/(?:shorts|embed|live)/([\w-]{11})", parsed.path)
            video_id = match.group(1) if match else None
    elif host == "youtu.be":
        video_id = parsed.path.lstrip("/")[:11]
    if not video_id or not re.fullmatch(r"[\w-]{11}", video_id):
        raise PipelineError("input", f"not a recognised YouTube video URL: {url!r}")
    return video_id


# ----------------------------------- Apify -----------------------------------------
def apify_run(actor: str, payload: dict, token: str, step: str, timeout_s: int = 280):
    """Run an allowed Apify actor synchronously and return its dataset items."""
    url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, params={"timeout": timeout_s}, json=payload,
                                 headers={"Authorization": f"Bearer {token}"},
                                 timeout=timeout_s + 40)
        except requests.RequestException as exc:
            last_error = f"network error: {type(exc).__name__}: {exc}"
        else:
            if resp.status_code in (200, 201):
                try:
                    return resp.json()
                except ValueError:
                    raise PipelineError(step, "Apify returned a non-JSON response.")
            try:
                err = resp.json()["error"]
                detail = f"{err.get('type', '')}: {err.get('message', '')}".strip()
                if err.get("approvalUrl"):
                    detail += (f" | One-time fix: approve this actor's permissions in "
                               f"Apify Console: {err['approvalUrl']}")
            except (ValueError, KeyError, TypeError, AttributeError):
                detail = resp.text[:300].replace("\n", " ")
            last_error = f"HTTP {resp.status_code}: {detail}"
            if resp.status_code not in RETRY_STATUS:
                raise PipelineError(step, last_error)        # permanent: do not retry
        if attempt < MAX_ATTEMPTS:
            wait = 2 * attempt
            log(f"{step}: transient failure ({last_error[:120]}); retry {attempt}/"
                f"{MAX_ATTEMPTS - 1} in {wait}s")
            time.sleep(wait)
    raise PipelineError(step, f"{last_error} (after {MAX_ATTEMPTS} attempts)")


def fetch_video_metadata(video_id: str, token: str) -> dict:
    items = apify_run(APIFY_YOUTUBE,
                      {"startUrls": [{"url": f"https://www.youtube.com/watch?v={video_id}"}]},
                      token, "youtube")
    if not items:
        raise PipelineError("youtube", "no data returned (private, removed or unavailable video?)")
    meta = items[0]
    if not (meta.get("title") or "").strip():
        raise PipelineError("youtube", "the scraper returned no video title")
    return meta


# ----------------------------------- LLM -------------------------------------------
ALBUM_SCHEMA = {
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
                    "properties": {"position": {"type": "integer"}, "title": {"type": "string"}},
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


def build_messages(meta: dict) -> list:
    description = (meta.get("text") or meta.get("description") or "")[:400].replace("\n", " ")
    user = USER_TEMPLATE.format(title=meta.get("title", ""),
                                channel=meta.get("channelName", ""), description=description)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _openrouter_post(payload: dict, key: str):
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(OPENROUTER_URL, json=payload, timeout=180,
                                 headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException as exc:
            last_error = f"network error: {type(exc).__name__}: {exc}"
        else:
            if resp.status_code not in RETRY_STATUS:
                return resp
            last_error = f"HTTP {resp.status_code}"
        if attempt < MAX_ATTEMPTS:
            wait = 2 * attempt
            log(f"llm: transient failure ({last_error}); retry {attempt}/{MAX_ATTEMPTS - 1} in {wait}s")
            time.sleep(wait)
    raise PipelineError("llm", f"{last_error} (after {MAX_ATTEMPTS} attempts)")


def call_openrouter(model: str, messages: list, key: str, max_tokens: int) -> dict:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "response_format": {"type": "json_schema", "json_schema": ALBUM_SCHEMA},
               "temperature": 0, "seed": LLM_SEED}
    resp = _openrouter_post(payload, key)
    if resp.status_code == 400:      # some models reject temperature/seed: retry once without
        log("llm: HTTP 400 with temperature/seed; retrying once without them")
        payload.pop("temperature"), payload.pop("seed")
        resp = _openrouter_post(payload, key)
    if resp.status_code != 200:
        hint = {401: " (invalid OPENROUTER_API_KEY)",
                402: " (insufficient OpenRouter credits: add credits or lower "
                     "OPENROUTER_MAX_TOKENS)"}.get(resp.status_code, "")
        raise PipelineError("llm", f"HTTP {resp.status_code}{hint}: {resp.text[:300]}")
    try:
        data = resp.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise PipelineError("llm", "answer was cut off (finish_reason=length); raise "
                                       "OPENROUTER_MAX_TOKENS")
        log(f"llm: served by {data.get('model')}")
        return json.loads(choice["message"]["content"])
    except PipelineError:
        raise
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise PipelineError("llm", f"unparseable response ({type(exc).__name__}): {resp.text[:200]}")


def validate_album(result: dict):
    """Return (artist, album_name, [clean track titles in release order]) or raise."""
    artist = (result.get("artist") or "").strip()
    album = (result.get("album_name") or "").strip()
    tracks = result.get("tracks") or []
    if not artist or not album:
        raise PipelineError("album", "the model returned an empty artist or album name")
    if not tracks:
        raise PipelineError("album", "the model returned an empty tracklist")
    ordered = sorted(tracks, key=lambda t: t.get("position", 0))
    if [t.get("position") for t in ordered] != list(range(1, len(ordered) + 1)):
        raise PipelineError("album", "track positions are not 1..N without gaps")
    titles = [clean_track_title(str(t.get("title", ""))) for t in ordered]
    if any(not t for t in titles):
        raise PipelineError("album", "a track title is empty")
    normalized = [_norm(t) for t in titles]
    if len(set(normalized)) != len(normalized):
        raise PipelineError("album", f"duplicate track titles: {titles}")
    excluded = {_norm(str(t)) for t in result.get("excluded_candidates", [])}
    if excluded & set(normalized):
        raise PipelineError("album", "a track is both included and excluded by the model")
    return artist, album, titles


def discover_album(meta: dict, model: str, key: str, max_tokens: int):
    return validate_album(call_openrouter(model, build_messages(meta), key, max_tokens))


# ----------------------------------- Genius ----------------------------------------
PAGE_FUNCTION = """
async function pageFunction(context) {
    const { request, response } = context;
    const containers = Array.from(document.querySelectorAll('[data-lyrics-container="true"]'));
    return {
        idx: request.userData.idx,
        requestedUrl: request.url,
        httpStatus: response ? response.status : null,
        title: document.title,
        innerTexts: containers.map(c => c.innerText)
    };
}
"""


def _slug(text: str) -> str:
    text = text.replace("&", " and ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"['`]", "", text)
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def genius_slug_url(artist: str, title: str) -> str:
    """A CANDIDATE url (e.g. https://genius.com/Madonna-holiday-lyrics), not proof of a match."""
    return f"https://genius.com/{_slug(artist)}-{_slug(title).lower()}-lyrics"


def validate_page(item: dict, artist: str, track: str):
    """Return (lyrics, None) for a verified page, else (None, reason)."""
    if item.get("httpStatus") != 200:
        return None, f"HTTP status {item.get('httpStatus')}"
    page_title = _norm(item.get("title") or "")
    if _norm(artist) not in page_title:
        return None, f"page title {item.get('title')!r} does not mention the artist"
    if _norm(track) not in page_title:
        return None, f"page title {item.get('title')!r} does not mention the track"
    extra = (set(page_title.split()) & VARIANT_WORDS) - set(_norm(track).split())
    if extra:
        return None, f"page looks like a different version ({sorted(extra)})"
    lyrics = "\n".join(item.get("innerTexts") or []).strip()
    if not lyrics:
        return None, "no lyrics text found on the page"
    return lyrics, None


def scrape_pages(pairs, token: str) -> list:
    """pairs: [(url, track_index)] -> dataset items, in ONE web-scraper run."""
    if not pairs:
        return []
    payload = {
        "startUrls": [{"url": u, "userData": {"idx": i}} for u, i in pairs],
        "pageFunction": PAGE_FUNCTION,
        "proxyConfiguration": {"useApifyProxy": True},
        "runMode": "PRODUCTION",
        "maxPagesPerCrawl": len(pairs),
        "maxRequestRetries": 1,
        "waitUntil": ["domcontentloaded"],   # lyrics are server-rendered; ads never go idle
        "pageLoadTimeoutSecs": 45,
        "pageFunctionTimeoutSecs": 30,
    }
    return apify_run(APIFY_WEB, payload, token, "genius-scrape")


def search_candidates(missing, artist: str, tracks, token: str) -> dict:
    """Fallback: Google search restricted to genius.com. Returns {track_index: [urls]}."""
    queries = {f"site:genius.com {artist} {tracks[i]} lyrics": i for i in missing}
    items = apify_run(APIFY_SEARCH, {"queries": "\n".join(queries), "resultsPerPage": 10,
                                     "maxPagesPerQuery": 1}, token, "genius-search")
    found = {}
    for item in items:
        term = ((item.get("searchQuery") or {}).get("term") or "").strip()
        idx = queries.get(term)
        if idx is None:
            continue
        urls = [r.get("url", "") for r in (item.get("organicResults") or [])]
        found.setdefault(idx, []).extend(u for u in urls if GENIUS_URL_RE.match(u))
    return found


def _log_unverified(missing, tracks, reasons) -> None:
    for i in missing:
        why = reasons.get(i, ["no page returned by the scraper (load failed or timed out)"])[-1]
        log(f"  not verified: {tracks[i]!r} -> {why}")


def fetch_all_lyrics(artist: str, tracks, token: str) -> list:
    """Verified lyrics for every track, in track order. Raises if any track cannot be found."""
    lyrics, reasons, tried = {}, {}, {i: [] for i in range(len(tracks))}

    def accept(items, rank):
        for item in sorted(items, key=lambda it: rank.get(it.get("idx"), {}).get(
                it.get("requestedUrl"), 99)):
            idx = item.get("idx")
            if idx is None or idx in lyrics or idx not in tried:
                continue
            text, reason = validate_page(item, artist, tracks[idx])
            if text is not None:
                lyrics[idx] = text
            else:
                reasons.setdefault(idx, []).append(f"{item.get('requestedUrl')}: {reason}")

    slug_pairs = [(genius_slug_url(artist, t), i) for i, t in enumerate(tracks)]
    for url, i in slug_pairs:
        tried[i].append(url)
    log(f"genius: scraping {len(slug_pairs)} candidate pages in one run")
    accept(scrape_pages(slug_pairs, token), {})

    missing = [i for i in range(len(tracks)) if i not in lyrics]
    if missing:
        log(f"genius: {len(missing)} track(s) not verified from the slug guess; "
            f"using Google search for: {[tracks[i] for i in missing]}")
        _log_unverified(missing, tracks, reasons)
        found = search_candidates(missing, artist, tracks, token)
        log(f"genius: Google returned Genius candidates for {len(found)} of {len(missing)} track(s)")
        pairs, rank = [], {}
        for i in missing:
            urls = []
            for u in found.get(i, []):
                if u not in tried[i] and u not in urls:
                    urls.append(u)
            for pos, u in enumerate(urls[:3]):
                pairs.append((u, i))
                tried[i].append(u)
                rank.setdefault(i, {})[u] = pos
        accept(scrape_pages(pairs, token), rank)

    missing = [i for i in range(len(tracks)) if i not in lyrics]
    if missing:
        _log_unverified(missing, tracks, reasons)
        detail = "; ".join(f"{tracks[i]!r} -> {reasons.get(i, ['no candidate page found'])[-1]}"
                           for i in missing)
        raise PipelineError("lyrics", f"no verified Genius lyrics for {len(missing)} track(s): {detail}")
    return [lyrics[i] for i in range(len(tracks))]


# ----------------------------------- orchestration ---------------------------------
def build_from_lyrics(artist: str, album: str, tracks, lyrics_list) -> dict:
    items = [{"track_index": i, "name": name, "metrics": lyrics_metrics(text)}
             for i, (name, text) in enumerate(zip(tracks, lyrics_list))]
    return build_analysis(artist, album, items)


def _cache_path(video_id: str, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    return CACHE_DIR / f"analysis_{video_id}_{PROMPT_VERSION}_{safe_model}_p{PIPELINE_VERSION}.json"


def _album_cache_path(video_id: str, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    return CACHE_DIR / f"album_{video_id}_{PROMPT_VERSION}_{safe_model}_p{PIPELINE_VERSION}.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log(f"could not write cache ({exc}); continuing")


def analyze_video(url: str, model: str, max_tokens: int, use_cache: bool) -> dict:
    video_id = parse_video_id(url)
    cache_file = _cache_path(video_id, model)
    if use_cache and cache_file.exists():
        try:
            analysis = json.loads(cache_file.read_text(encoding="utf-8"))
            log(f"using cached analysis ({cache_file.name}); pass --no-cache to recompute")
            return analysis
        except (OSError, ValueError):
            log("cache file unreadable; recomputing")

    apify_key = require_env("APIFY_API_KEY")
    album_file = _album_cache_path(video_id, model)
    cached = _read_json(album_file) if use_cache else None
    if (isinstance(cached, dict) and cached.get("artist") and cached.get("album")
            and isinstance(cached.get("tracks"), list) and cached["tracks"]):
        artist, album, tracks = cached["artist"], cached["album"], cached["tracks"]
        log(f"steps 1-2/4: using cached album ({album_file.name}): {artist} - {album}, "
            f"{len(tracks)} tracks")
    else:
        openrouter_key = require_env("OPENROUTER_API_KEY")
        log("step 1/4: YouTube metadata")
        meta = fetch_video_metadata(video_id, apify_key)
        log(f"  video: {meta.get('title')!r}")
        log(f"step 2/4: first album via {model}")
        artist, album, tracks = discover_album(meta, model, openrouter_key, max_tokens)
        log(f"  {artist} - {album}: {len(tracks)} tracks")
        if use_cache:
            _write_json(album_file, {"artist": artist, "album": album, "tracks": tracks})
    log("step 3/4: Genius lyrics")
    lyrics_list = fetch_all_lyrics(artist, tracks, apify_key)
    log("step 4/4: metrics")
    analysis = build_from_lyrics(artist, album, tracks, lyrics_list)

    if use_cache:
        _write_json(cache_file, analysis)
    return analysis


def run(url: str, mode: str, model: str, max_tokens: int, use_cache: bool) -> str:
    analysis = analyze_video(url, model, max_tokens, use_cache)
    if mode == "json":
        return json.dumps(analysis, ensure_ascii=False, indent=2)
    token_string = token_count_string(analysis)
    log(f"embedding input: {token_string}")
    try:
        return compute_embedding_hash(token_string)
    except Exception as exc:  # noqa: BLE001 - reported with its step
        raise PipelineError("embedding", f"{type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="YouTube AI Scraping Agent: song URL -> first album -> Genius lyrics metrics.")
    parser.add_argument("url", help="YouTube video URL of a song")
    parser.add_argument("mode", choices=["json", "hash"],
                        help="json: analysis JSON; hash: final embedding MD5")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")
    args = parser.parse_args(argv)

    load_env()
    model = os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
    try:
        max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "") or DEFAULT_MAX_TOKENS)
    except ValueError:
        print("[ERROR] config: OPENROUTER_MAX_TOKENS must be an integer", file=sys.stderr)
        return 1

    real_stdout = sys.stdout
    try:
        # Libraries (model loading) may print to stdout: keep stdout clean for the result.
        with contextlib.redirect_stdout(sys.stderr):
            output = run(args.url, args.mode, model, max_tokens, not args.no_cache)
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(output, file=real_stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())