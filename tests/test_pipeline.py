"""Offline tests for the pipeline pieces (mocked Apify/OpenRouter; no network, no credits)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import Seda_Nur_Kilic_YouTube_AI as core  # noqa: E402


class FakeEncoder:
    def encode(self, text, **kwargs):
        return text.split()


class TestVideoId(unittest.TestCase):
    def test_accepted_formats(self):
        for url in ["https://www.youtube.com/watch?v=rSaC-YbSDpo",
                    "https://youtube.com/watch?v=rSaC-YbSDpo&t=5",
                    "https://m.youtube.com/watch?v=rSaC-YbSDpo",
                    "https://youtu.be/rSaC-YbSDpo",
                    "https://www.youtube.com/shorts/rSaC-YbSDpo"]:
            with self.subTest(url=url):
                self.assertEqual(core.parse_video_id(url), "rSaC-YbSDpo")

    def test_rejected(self):
        for url in ["", "hello", "https://example.com/watch?v=rSaC-YbSDpo",
                    "https://www.youtube.com/watch?v=short", "https://www.youtube.com/"]:
            with self.subTest(url=url):
                with self.assertRaises(core.PipelineError):
                    core.parse_video_id(url)


class TestSlug(unittest.TestCase):
    def test_slugs(self):
        cases = {("Madonna", "Holiday"): "https://genius.com/Madonna-holiday-lyrics",
                 ("Madonna", "I Know It"): "https://genius.com/Madonna-i-know-it-lyrics",
                 ("Rihanna", "You Don't Love Me (No, No, No)"):
                     "https://genius.com/Rihanna-you-dont-love-me-no-no-no-lyrics",
                 ("Hall & Oates", "Maneater"): "https://genius.com/Hall-and-Oates-maneater-lyrics",
                 ("Beyoncé", "Halo"): "https://genius.com/Beyonce-halo-lyrics"}
        for (artist, title), expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(core.genius_slug_url(artist, title), expected)


class TestValidatePage(unittest.TestCase):
    def item(self, **kw):
        base = {"httpStatus": 200, "title": "Madonna – Holiday Lyrics | Genius Lyrics",
                "innerTexts": ["[Verse]\nla la", "", "na na"], "requestedUrl": "u", "idx": 0}
        base.update(kw)
        return base

    def test_good_page(self):
        lyrics, reason = core.validate_page(self.item(), "Madonna", "Holiday")
        self.assertIsNone(reason)
        self.assertEqual(lyrics, "[Verse]\nla la\n\nna na")      # P2: joined with \n, trimmed

    def test_rejections(self):
        cases = [self.item(httpStatus=404),
                 self.item(title="Someone Else – Holiday Lyrics | Genius Lyrics"),
                 self.item(title="Madonna – Other Song Lyrics | Genius Lyrics"),
                 self.item(title="Madonna – Holiday (Remix) Lyrics | Genius Lyrics"),
                 self.item(title="Madonna – Holiday (Live) Lyrics | Genius Lyrics"),
                 self.item(innerTexts=["", " "])]
        for it in cases:
            with self.subTest(title=it["title"]):
                lyrics, reason = core.validate_page(it, "Madonna", "Holiday")
                self.assertIsNone(lyrics)
                self.assertTrue(reason)

    def test_variant_word_allowed_when_the_track_has_it(self):
        it = self.item(title="Madonna – Holiday (Remix) Lyrics | Genius Lyrics")
        self.assertIsNotNone(core.validate_page(it, "Madonna", "Holiday (Remix)")[0])

    def test_word_boundaries_not_substrings(self):
        it = self.item(title="Madonna – Alive Holiday Lyrics | Genius Lyrics")   # 'alive' != 'live'
        self.assertIsNotNone(core.validate_page(it, "Madonna", "Holiday")[0])


class TestValidateAlbum(unittest.TestCase):
    def good(self, **kw):
        base = {"artist": "A", "album_name": "X", "release_year": 1990,
                "tracks": [{"position": 2, "title": "Two (feat. Z)"}, {"position": 1, "title": "One"}],
                "excluded_candidates": ["Bonus"], "notes": ""}
        base.update(kw)
        return base

    def test_orders_by_position_and_cleans_titles(self):
        self.assertEqual(core.validate_album(self.good()), ("A", "X", ["One", "Two"]))

    def test_failures(self):
        bad = [self.good(artist=""), self.good(tracks=[]),
               self.good(tracks=[{"position": 1, "title": "One"}, {"position": 3, "title": "Two"}]),
               self.good(tracks=[{"position": 1, "title": "One"}, {"position": 2, "title": "one"}]),
               self.good(excluded_candidates=["two"])]
        for result in bad:
            with self.subTest(result=result):
                with self.assertRaises(core.PipelineError):
                    core.validate_album(result)


class TestFetchAllLyrics(unittest.TestCase):
    def page(self, idx, url, artist="A", track="One", status=200, text="la la"):
        return {"idx": idx, "requestedUrl": url, "httpStatus": status,
                "title": f"{artist} – {track} Lyrics | Genius Lyrics", "innerTexts": [text]}

    def test_slug_hits_need_one_scrape_and_no_search(self):
        tracks = ["One", "Two"]
        items = [self.page(0, "u0", track="One"), self.page(1, "u1", track="Two")]
        with mock.patch.object(core, "scrape_pages", return_value=items) as scrape, \
             mock.patch.object(core, "search_candidates") as search, \
             contextlib.redirect_stderr(io.StringIO()):
            out = core.fetch_all_lyrics("A", tracks, "t")
        self.assertEqual(out, ["la la", "la la"])
        self.assertEqual(scrape.call_count, 1)
        search.assert_not_called()

    def test_fallback_uses_search_and_keeps_track_order(self):
        tracks = ["One", "Two"]
        first = [self.page(1, "u1", track="Two", text="second")]                 # track 0 missing
        second = [self.page(0, "https://genius.com/A-one-live-lyrics", track="One (Live)", text="bad"),
                  self.page(0, "https://genius.com/A-one-lyrics", track="One", text="first")]
        found = {0: ["https://genius.com/A-one-live-lyrics", "https://genius.com/A-one-lyrics"]}
        with mock.patch.object(core, "scrape_pages", side_effect=[first, second]), \
             mock.patch.object(core, "search_candidates", return_value=found) as search, \
             contextlib.redirect_stderr(io.StringIO()):
            out = core.fetch_all_lyrics("A", tracks, "t")
        self.assertEqual(out, ["first", "second"])
        self.assertEqual(search.call_args.args[0], [0])

    def test_missing_track_raises_instead_of_being_dropped(self):
        with mock.patch.object(core, "scrape_pages", side_effect=[[], []]), \
             mock.patch.object(core, "search_candidates", return_value={}), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(core.PipelineError) as ctx:
                core.fetch_all_lyrics("A", ["One"], "t")
        self.assertIn("One", str(ctx.exception))

    def test_rejection_reasons_are_logged(self):
        wrong = [self.page(0, "u0", track="Different Song")]
        err = io.StringIO()
        with mock.patch.object(core, "scrape_pages", side_effect=[wrong, []]), \
             mock.patch.object(core, "search_candidates", return_value={}), \
             contextlib.redirect_stderr(err):
            with self.assertRaises(core.PipelineError):
                core.fetch_all_lyrics("A", ["One"], "t")
        self.assertIn("not verified: 'One'", err.getvalue())
        self.assertIn("does not mention the track", err.getvalue())

    def test_scraper_input_does_not_wait_for_network_idle(self):
        with mock.patch.object(core, "apify_run", return_value=[]) as run:
            core.scrape_pages([("https://genius.com/A-one-lyrics", 0)], "t")
        payload = run.call_args.args[1]
        self.assertEqual(payload["waitUntil"], ["domcontentloaded"])
        self.assertEqual(payload["startUrls"][0]["userData"], {"idx": 0})

    def test_wrong_song_page_is_rejected(self):
        wrong = [self.page(0, "u0", track="Different Song")]
        with mock.patch.object(core, "scrape_pages", side_effect=[wrong, []]), \
             mock.patch.object(core, "search_candidates", return_value={}), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(core.PipelineError):
                core.fetch_all_lyrics("A", ["One"], "t")


class TestCli(unittest.TestCase):
    META = {"title": "A - One (Official Video)", "channelName": "A", "text": "d"}
    ALBUM = ("A", "X", ["One", "Two"])
    LYRICS = ["la la la", "na na na na"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [mock.patch.object(core, "CACHE_DIR", Path(self.tmp.name)),
                   mock.patch.object(core, "load_env"),
                   mock.patch.object(core, "require_env", return_value="k"),
                   mock.patch.object(core, "get_encoder", return_value=FakeEncoder()),
                   mock.patch.object(core, "fetch_video_metadata", return_value=self.META),
                   mock.patch.object(core, "discover_album", return_value=self.ALBUM),
                   mock.patch.object(core, "fetch_all_lyrics", return_value=self.LYRICS)]
        self.mocks = [p.start() for p in patches]
        self.addCleanup(lambda: [p.stop() for p in patches])
        self.addCleanup(self.tmp.cleanup)

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = core.main(list(args))
        return code, out.getvalue(), err.getvalue()

    URL = "https://www.youtube.com/watch?v=rSaC-YbSDpo"

    def test_json_mode_prints_only_the_json(self):
        code, out, _ = self.cli(self.URL, "json", "--no-cache")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(list(data), ["artist", "album_name", "songs",
                                      "total_tokens_all_songs", "avg_tokens_per_song"])
        self.assertEqual([s["name"] for s in data["songs"]], ["One", "Two"])
        self.assertEqual([s["lyrics_length_tokens"] for s in data["songs"]], [3, 4])
        self.assertEqual(data["total_tokens_all_songs"], 7)
        self.assertEqual(data["avg_tokens_per_song"], 3.5)

    def test_hash_mode_prints_only_the_hash_even_if_libraries_print(self):
        seen = {}

        def noisy(token_string):
            seen["input"] = token_string
            print("<All keys matched successfully>")      # what the model loader does
            return "a" * 32

        with mock.patch.object(core, "compute_embedding_hash", noisy):
            code, out, err = self.cli(self.URL, "hash", "--no-cache")
        self.assertEqual(code, 0)
        self.assertEqual(out, "a" * 32 + "\n")
        self.assertEqual(seen["input"], "3,4")
        self.assertIn("All keys matched", err)

    def test_failure_prints_error_to_stderr_and_nothing_to_stdout(self):
        with mock.patch.object(core, "fetch_video_metadata",
                               side_effect=core.PipelineError("youtube", "boom")):
            code, out, err = self.cli(self.URL, "json", "--no-cache")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("[ERROR] youtube: boom", err)

    def test_invalid_url_is_a_clean_error(self):
        code, out, err = self.cli("not a url", "json")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("[ERROR] input:", err)

    def test_embedding_failure_is_reported_with_its_step(self):
        with mock.patch.object(core, "compute_embedding_hash", side_effect=RuntimeError("no model")):
            code, out, err = self.cli(self.URL, "hash", "--no-cache")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("[ERROR] embedding: RuntimeError: no model", err)

    def test_second_run_uses_the_cache_and_gives_identical_output(self):
        _, first, _ = self.cli(self.URL, "json")
        self.mocks[4].side_effect = AssertionError("must not call YouTube again")
        code, second, err = self.cli(self.URL, "json")
        self.assertEqual(code, 0)
        self.assertEqual(first, second)
        self.assertIn("cached", err)

    def test_album_cache_skips_youtube_and_llm_after_a_later_failure(self):
        # lyrics step fails the first time: the album must already be cached
        with mock.patch.object(core, "fetch_all_lyrics",
                               side_effect=core.PipelineError("lyrics", "boom")):
            code, _, _ = self.cli(self.URL, "json")
        self.assertEqual(code, 1)
        self.mocks[4].side_effect = AssertionError("YouTube must not be called again")
        self.mocks[5].side_effect = AssertionError("LLM must not be called again")
        code, out, err = self.cli(self.URL, "json")
        self.assertEqual(code, 0)
        self.assertIn("cached album", err)
        self.assertEqual([s["name"] for s in json.loads(out)["songs"]], ["One", "Two"])

    def test_no_cache_flag_ignores_the_album_cache(self):
        self.cli(self.URL, "json")
        self.mocks[5].reset_mock()
        self.cli(self.URL, "json", "--no-cache")
        self.mocks[5].assert_called_once()

    def test_cache_files_contain_no_lyrics(self):
        self.cli(self.URL, "json")
        text = "".join(p.read_text() for p in Path(self.tmp.name).glob("*.json"))
        self.assertNotIn("la la la", text)

    def test_invalid_mode_is_rejected_by_argparse(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                core.main([self.URL, "xml"])


class TestOpenRouterErrors(unittest.TestCase):
    def resp(self, code, body):
        r = mock.Mock(status_code=code, text=json.dumps(body))
        r.json.return_value = body
        return r

    def test_402_gives_an_actionable_message(self):
        with mock.patch.object(core.requests, "post", return_value=self.resp(402, {"error": "x"})):
            with self.assertRaises(core.PipelineError) as ctx:
                core.call_openrouter("m", [], "k", 100)
        self.assertIn("credits", str(ctx.exception))

    def test_truncated_answer_is_an_error(self):
        body = {"model": "m", "choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        with mock.patch.object(core.requests, "post", return_value=self.resp(200, body)):
            with self.assertRaises(core.PipelineError):
                core.call_openrouter("m", [], "k", 100)

    def test_transient_error_is_retried_then_succeeds(self):
        good = {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": '{"a": 1}'}}]}
        with mock.patch.object(core.requests, "post",
                               side_effect=[self.resp(503, {}), self.resp(200, good)]) as post, \
             mock.patch.object(core.time, "sleep"), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(core.call_openrouter("m", [], "k", 100), {"a": 1})
        self.assertEqual(post.call_count, 2)


class TestApifyErrors(unittest.TestCase):
    def test_permission_error_shows_the_approval_link_and_is_not_retried(self):
        r = mock.Mock(status_code=403, text="")
        r.json.return_value = {"error": {"type": "full-permission-actor-not-approved",
                                         "message": "approve", "approvalUrl": "https://console.apify.com/x"}}
        with mock.patch.object(core.requests, "post", return_value=r) as post:
            with self.assertRaises(core.PipelineError) as ctx:
                core.apify_run("a~b", {}, "t", "genius-scrape")
        self.assertIn("https://console.apify.com/x", str(ctx.exception))
        self.assertEqual(post.call_count, 1)

    def test_network_errors_are_retried_a_bounded_number_of_times(self):
        with mock.patch.object(core.requests, "post",
                               side_effect=core.requests.ConnectionError("down")) as post, \
             mock.patch.object(core.time, "sleep"), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(core.PipelineError):
                core.apify_run("a~b", {}, "t", "youtube")
        self.assertEqual(post.call_count, core.MAX_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()