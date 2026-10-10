"""
Tests for the deterministic metrics core. Only INVENTED texts are used here
(no copyrighted lyrics). Run:  python -m unittest discover -s tests -v

Two layers:
  * logic tests   - always run (tokenizer replaced by a stub)
  * real-token tests - need tiktoken's cl100k_base file; skipped with a visible
                       reason if it cannot be loaded (e.g. offline)
"""

import hashlib
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import Seda_Nur_Kilic_YouTube_AI as core  # noqa: E402


class FakeEncoder:
    """Stand-in tokenizer: one token per whitespace-separated piece."""

    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append(kwargs)
        return text.split()


# --------------------------------- characters --------------------------------------
class TestChars(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(core.count_chars("abc"), 3)

    def test_newlines_count_as_characters(self):
        self.assertEqual(core.count_chars("a\nb"), 3)

    def test_no_unicode_normalization(self):
        precomposed = "\u00e9"        # 'é' as one code point
        decomposed = "e\u0301"        # 'e' + combining accent
        self.assertEqual(core.count_chars(precomposed), 1)
        self.assertEqual(core.count_chars(decomposed), 2)

    def test_emoji_is_one_code_point(self):
        self.assertEqual(core.count_chars("\U0001F600"), 1)


# ---------------------------------- words ------------------------------------------
class TestWords(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(core.count_words("one two three"), 3)

    def test_runs_of_whitespace_and_newlines(self):
        self.assertEqual(core.count_words("one   two\n\nthree\t four "), 4)

    def test_apostrophe_stays_inside_word(self):
        self.assertEqual(core.count_words("don't stop"), 2)

    def test_section_tag_counts_as_a_word_under_split_rule(self):
        # Documents the PROVISIONAL rule: '[Chorus]' is one whitespace-delimited word.
        self.assertEqual(core.count_words("[Chorus]\nla la"), 3)

    def test_lone_dash_counts_as_a_word_under_split_rule(self):
        self.assertEqual(core.count_words("la - la"), 3)


# ------------------------------------ MD5 ------------------------------------------
class TestLyricsMd5(unittest.TestCase):
    def test_known_vectors(self):
        self.assertEqual(core.lyrics_md5("abc"), "900150983cd24fb0d6963f7d28e17f72")
        self.assertEqual(core.lyrics_md5(""), "d41d8cd98f00b204e9800998ecf8427e")

    def test_uses_utf8_bytes(self):
        text = "caf\u00e9"
        self.assertEqual(core.lyrics_md5(text), hashlib.md5(text.encode("utf-8")).hexdigest())
        self.assertNotEqual(core.lyrics_md5(text), hashlib.md5(text.encode("latin-1")).hexdigest())

    def test_newline_style_changes_hash(self):
        self.assertNotEqual(core.lyrics_md5("a\nb"), core.lyrics_md5("a\r\nb"))

    def test_format(self):
        digest = core.lyrics_md5("anything")
        self.assertEqual(len(digest), 32)
        self.assertEqual(digest, digest.lower())


# ------------------------- tokens (logic, stub tokenizer) ---------------------------
class TestTokenLogic(unittest.TestCase):
    def test_count_is_length_of_encoded_list(self):
        with mock.patch.object(core, "get_encoder", return_value=FakeEncoder()):
            self.assertEqual(core.count_tokens("a b c d"), 4)

    def test_special_token_strings_are_not_disallowed(self):
        enc = FakeEncoder()
        with mock.patch.object(core, "get_encoder", return_value=enc):
            core.count_tokens("<|endoftext|> hi")
        self.assertEqual(enc.calls, [{"disallowed_special": ()}])

    def test_lyrics_metrics_uses_one_string_for_everything(self):
        text = "la la\nla"
        with mock.patch.object(core, "get_encoder", return_value=FakeEncoder()):
            m = core.lyrics_metrics(text)
        self.assertEqual(m["lyrics_length_chars"], len(text))
        self.assertEqual(m["lyrics_length_words"], 3)
        self.assertEqual(m["lyrics_length_tokens"], 3)
        self.assertEqual(m["tokens_per_word"], 1.0)
        self.assertEqual(m["lyrics_hash"], hashlib.md5(text.encode("utf-8")).hexdigest())
        self.assertEqual(list(m), list(core.SONG_METRIC_KEYS))

    def test_zero_word_inputs_are_rejected(self):
        for bad in ("", "   ", "\n\t \n"):
            with self.subTest(bad=repr(bad)):
                with mock.patch.object(core, "get_encoder", return_value=FakeEncoder()):
                    with self.assertRaises(ValueError):
                        core.lyrics_metrics(bad)

    def test_non_string_is_rejected(self):
        with self.assertRaises(TypeError):
            core.lyrics_metrics(None)


# ---------------------- tokens (real cl100k_base, may be skipped) -------------------
class TestRealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            core.get_encoder.cache_clear()
            core.get_encoder()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"cl100k_base could not be loaded ({type(exc).__name__})")

    def test_known_counts(self):
        self.assertEqual(core.count_tokens(""), 0)
        self.assertEqual(core.count_tokens("hello world"), 2)
        self.assertEqual(core.count_tokens("tiktoken is great!"), 6)

    def test_special_token_text_does_not_raise(self):
        self.assertGreater(core.count_tokens("<|endoftext|>"), 0)

    def test_unicode_and_newlines_are_countable(self):
        self.assertGreater(core.count_tokens("caf\u00e9\n\U0001F600"), 0)

    def test_repeatable(self):
        text = "la la la\nna na"
        self.assertEqual(core.count_tokens(text), core.count_tokens(text))


# ------------------------- tokens-per-word and rounding -----------------------------
class TestRatioAndRounding(unittest.TestCase):
    def test_reference_values_from_assessment_example(self):
        # (tokens, words, expected) taken from the assessment's example JSON.
        for tokens, words, expected in [(395, 264, 1.5), (703, 429, 1.64),
                                        (470, 349, 1.35), (634, 410, 1.55)]:
            with self.subTest(tokens=tokens, words=words):
                self.assertEqual(core.tokens_per_word(tokens, words), expected)

    def test_zero_words_error(self):
        with self.assertRaises(ValueError):
            core.tokens_per_word(10, 0)

    def test_rounding_policy_is_python_round_half_even_on_exact_binary(self):
        # Exact binary ties go to the even neighbour...
        self.assertEqual(round(630.875, 2), 630.88)   # matches the example's average
        self.assertEqual(round(2.125, 2), 2.12)
        self.assertEqual(round(0.125, 2), 0.12)
        # ...and decimals that are slightly below/above the tie follow the stored value.
        self.assertEqual(round(1.005, 2), 1.0)


# ---------------------------------- public JSON -------------------------------------
REFERENCE_TOKENS = [395, 703, 470, 622, 561, 950, 712, 634]


def fake_tracks():
    """Eight tracks whose token counts equal the assessment example (invented names)."""
    tracks = []
    for i, tokens in enumerate(REFERENCE_TOKENS):
        tracks.append({
            "track_index": i,
            "name": f"Track {i + 1}",
            "metrics": {
                "lyrics_length_chars": 1000 + i,
                "lyrics_length_words": 200 + i,
                "lyrics_length_tokens": tokens,
                "tokens_per_word": round(tokens / (200 + i), 2),
                "lyrics_hash": f"{i:032x}",
            },
        })
    return tracks


class TestBuildAnalysis(unittest.TestCase):
    def test_totals_average_and_token_string(self):
        result = core.build_analysis("Artist", "Album", fake_tracks())
        self.assertEqual(result["total_tokens_all_songs"], 5047)
        self.assertEqual(result["avg_tokens_per_song"], 630.88)
        self.assertEqual(core.token_count_string(result), "395,703,470,622,561,950,712,634")

    def test_order_follows_track_index_not_completion_order(self):
        shuffled = fake_tracks()
        shuffled.reverse()
        shuffled[0], shuffled[3] = shuffled[3], shuffled[0]
        result = core.build_analysis("Artist", "Album", shuffled)
        self.assertEqual([s["name"] for s in result["songs"]],
                         [f"Track {i}" for i in range(1, 9)])

    def test_public_json_has_exactly_the_specified_fields(self):
        result = core.build_analysis("Artist", "Album", fake_tracks())
        self.assertEqual(list(result), ["artist", "album_name", "songs",
                                        "total_tokens_all_songs", "avg_tokens_per_song"])
        self.assertEqual(list(result["songs"][0]),
                         ["name", "lyrics_length_chars", "lyrics_length_words",
                          "lyrics_length_tokens", "tokens_per_word", "lyrics_hash"])
        self.assertNotIn("track_index", result["songs"][0])

    def test_is_json_serializable(self):
        result = core.build_analysis("Artist", "Album", fake_tracks())
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_duplicate_index_rejected(self):
        tracks = fake_tracks()
        tracks[1]["track_index"] = 0
        with self.assertRaises(ValueError):
            core.build_analysis("Artist", "Album", tracks)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            core.build_analysis("Artist", "Album", [])


if __name__ == "__main__":
    unittest.main()
