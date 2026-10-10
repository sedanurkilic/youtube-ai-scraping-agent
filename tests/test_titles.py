"""Tests for clean_track_title(): deterministic removal of trailing featured-artist credits."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import Seda_Nur_Kilic_YouTube_AI as core  # noqa: E402


class TestCleanTrackTitle(unittest.TestCase):
    def test_trailing_feature_credits_are_removed(self):
        cases = {
            "There's a Thug in My Life (feat. J-Status)": "There's a Thug in My Life",
            "Song (Feat. A & B)": "Song",
            "Song [ft. A]": "Song",
            "Song (ft A)": "Song",
            "Song (featuring A)": "Song",
            "Song - feat. A": "Song",
            "  Song (feat. A)  ": "Song",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(core.clean_track_title(raw), expected)

    def test_real_title_parts_are_left_alone(self):
        for title in ["You Don't Love Me (No, No, No)", "Holiday (Remix)", "Song (with A)",
                      "Feat of Strength", "Left Outside Alone", "Plain Title", "Song (Live)"]:
            with self.subTest(title=title):
                self.assertEqual(core.clean_track_title(title), title)

    def test_never_returns_an_empty_title(self):
        self.assertEqual(core.clean_track_title("(feat. X)"), "(feat. X)")
        self.assertEqual(core.clean_track_title("   "), "")

    def test_only_a_trailing_credit_is_removed(self):
        self.assertEqual(core.clean_track_title("A (feat. B) Reprise"), "A (feat. B) Reprise")

    def test_idempotent(self):
        once = core.clean_track_title("Song (feat. A)")
        self.assertEqual(core.clean_track_title(once), once)


if __name__ == "__main__":
    unittest.main()
