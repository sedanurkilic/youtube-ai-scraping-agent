"""Offline tests for the golden-set comparison logic (no network, no credits)."""

import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval_golden as ev  # noqa: E402

MADONNA = ["Lucky Star", "Borderline", "Burning Up", "I Know It", "Holiday",
           "Think of Me", "Physical Attraction", "Everybody"]
RIHANNA_REQ = ["Pon de Replay", "Here I Go Again", "If It's Lovin' That You Want",
               "You Don't Love Me (No, No, No)", "That La, La, La", "The Last Time",
               "Willing to Wait", "Music of the Sun", "Let Me", "Rush",
               "There's a Thug in My Life"]


class TestNorm(unittest.TestCase):
    def test_case_and_punctuation_are_ignored(self):
        self.assertEqual(ev.norm("You Don't Love Me (No,No,No)"),
                         ev.norm("you don\u2019t love me (No, No, No)"))
        self.assertEqual(ev.norm("Pon De Replay"), ev.norm("Pon de Replay"))

    def test_different_titles_stay_different(self):
        self.assertNotEqual(ev.norm("Holiday"), ev.norm("Holiday (Remix)"))


class TestCompareTracklist(unittest.TestCase):
    def test_exact_match_passes(self):
        self.assertTrue(ev.compare_tracklist(MADONNA, [], MADONNA)["ok"])

    def test_prompt_v1_regression_extra_stay_track_fails(self):
        cmp = ev.compare_tracklist(MADONNA, [], MADONNA + ["Stay"])
        self.assertFalse(cmp["ok"])
        self.assertEqual(cmp["extra"], ["Stay"])
        self.assertIn("Stay", ev.describe_comparison(cmp))

    def test_missing_track_fails(self):
        cmp = ev.compare_tracklist(MADONNA, [], MADONNA[:-1])
        self.assertFalse(cmp["ok"])
        self.assertEqual(cmp["missing"], ["Everybody"])

    def test_wrong_order_fails(self):
        swapped = [MADONNA[1], MADONNA[0]] + MADONNA[2:]
        cmp = ev.compare_tracklist(MADONNA, [], swapped)
        self.assertFalse(cmp["ok"])
        self.assertFalse(cmp["order_ok"])

    def test_duplicate_track_fails(self):
        self.assertFalse(ev.compare_tracklist(MADONNA, [], MADONNA + ["Holiday"])["ok"])

    def test_title_formatting_differences_are_tolerated(self):
        loose = [t.upper() for t in MADONNA]
        self.assertTrue(ev.compare_tracklist(MADONNA, [], loose)["ok"])

    def test_optional_track_at_the_end_is_accepted(self):
        cmp = ev.compare_tracklist(RIHANNA_REQ, ["Now I Know"], RIHANNA_REQ + ["Now I Know"])
        self.assertTrue(cmp["ok"])
        self.assertEqual(cmp["optional_used"], ["Now I Know"])

    def test_answer_without_the_optional_track_is_accepted(self):
        self.assertTrue(ev.compare_tracklist(RIHANNA_REQ, ["Now I Know"], RIHANNA_REQ)["ok"])

    def test_optional_track_in_the_middle_fails(self):
        misplaced = RIHANNA_REQ[:3] + ["Now I Know"] + RIHANNA_REQ[3:]
        cmp = ev.compare_tracklist(RIHANNA_REQ, ["Now I Know"], misplaced)
        self.assertFalse(cmp["ok"])
        self.assertFalse(cmp["optional_after_required"])

    def test_remix_bonus_track_fails_even_when_optional_is_allowed(self):
        answer = RIHANNA_REQ + ["Now I Know", "Pon de Replay (Remix)"]
        cmp = ev.compare_tracklist(RIHANNA_REQ, ["Now I Know"], answer)
        self.assertFalse(cmp["ok"])
        self.assertEqual(cmp["extra"], ["Pon de Replay (Remix)"])


class TestGoldenFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "golden_albums.json")
        with open(path, encoding="utf-8") as fh:
            cls.cases = json.load(fh)["cases"]

    def test_every_case_is_well_formed(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                for key in ("id", "url", "expect_title_contains", "artist", "album",
                            "required", "optional", "verified_with"):
                    self.assertIn(key, case)
                self.assertGreaterEqual(len(case["required"]), 5)
                self.assertEqual(len(ev.video_id(case["url"])), 11)
                norms = [ev.norm(t) for t in case["required"] + case["optional"]]
                self.assertEqual(len(norms), len(set(norms)), "duplicate titles in golden data")

    def test_assessment_case_is_present(self):
        self.assertIn("rSaC-YbSDpo", [ev.video_id(c["url"]) for c in self.cases])


class TestVideoId(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(ev.video_id("https://www.youtube.com/watch?v=abc123DEF45&t=3"), "abc123DEF45")
        self.assertEqual(ev.video_id("https://youtu.be/abc123DEF45"), "abc123DEF45")


class TestRunCaseWithMocks(unittest.TestCase):
    CASE = {"id": "t", "url": "https://www.youtube.com/watch?v=AAAAAAAAAAA",
            "expect_title_contains": "Song", "artist": "A", "album": "X",
            "required": ["One", "Two"], "optional": []}

    def answer(self, titles):
        return {"artist": "A", "album_name": "X", "release_year": 1990,
                "tracks": [{"position": i + 1, "title": t} for i, t in enumerate(titles)],
                "excluded_candidates": [], "notes": ""}

    def run_case(self, meta, answers):
        with mock.patch.object(ev, "get_metadata", return_value=(meta, "cache", None)), \
             mock.patch.object(ev, "ask_llm", side_effect=[(a, None) for a in answers]), \
             contextlib.redirect_stdout(io.StringIO()):
            return ev.run_case(self.CASE, len(answers), "m", 100, "t", "k")

    def test_pass(self):
        out = self.run_case({"title": "A - Song"}, [self.answer(["One", "Two"])] * 2)
        self.assertEqual(out, "PASS")

    def test_wrong_video_fails_without_calling_the_llm(self):
        with mock.patch.object(ev, "get_metadata", return_value=({"title": "Other"}, "cache", None)), \
             mock.patch.object(ev, "ask_llm") as ask, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ev.run_case(self.CASE, 2, "m", 100, "t", "k"), "FAIL")
            ask.assert_not_called()

    def test_inconsistent_runs_fail_even_if_each_is_valid(self):
        out = self.run_case({"title": "A - Song"},
                            [self.answer(["One", "Two"]), self.answer(["One", "Two"]) | {"release_year": 1991}])
        self.assertEqual(out, "FAIL")

    def test_http_402_stops_the_evaluation_with_a_clear_exception(self):
        err = "HTTP 402 insufficient OpenRouter credits: ..."
        with mock.patch.object(ev, "get_metadata", return_value=({"title": "A - Song"}, "cache", None)), \
             mock.patch.object(ev, "ask_llm", return_value=(None, err)), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ev.OutOfCredits):
                ev.run_case(self.CASE, 2, "m", 100, "t", "k")

    def test_other_llm_errors_are_reported_not_raised(self):
        with mock.patch.object(ev, "get_metadata", return_value=({"title": "A - Song"}, "cache", None)), \
             mock.patch.object(ev, "ask_llm", return_value=(None, "network error: x")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ev.run_case(self.CASE, 2, "m", 100, "t", "k"), "ERROR")

    def test_featuring_credit_variants_are_cleaned_and_count_as_consistent(self):
        plain = self.answer(["One", "Two"])
        credited = self.answer(["One", "Two (feat. Someone)"])
        out = self.run_case({"title": "A - Song"}, [plain, credited])
        self.assertEqual(out, "PASS")

    def test_extra_track_fails(self):
        out = self.run_case({"title": "A - Song"}, [self.answer(["One", "Two", "Three"])])
        self.assertEqual(out, "FAIL")


if __name__ == "__main__":
    unittest.main()