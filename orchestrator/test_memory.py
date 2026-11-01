"""Unit tests for conversational memory (no GPU / no live LLM)."""

import json
import tempfile
import unittest
from pathlib import Path

from memory import (
    MemoryStore,
    lexical_score,
    parse_fact_payload,
    tokens,
)


class FakeLLM:
    def __init__(self, payload="[]"):
        self.payload = payload
        self.calls = []

    def complete(self, system, user, max_tokens=None):
        self.calls.append((system, user, max_tokens))
        return self.payload


class TokenScoreTest(unittest.TestCase):
    def test_tokens_drop_stopwords(self):
        self.assertIn("coffee", tokens("I would like coffee please"))
        self.assertNotIn("would", tokens("I would like coffee please"))

    def test_score_ranks_overlap(self):
        q = tokens("user prefers dark roast coffee")
        hi = lexical_score(q, "The user prefers dark roast coffee in the morning")
        lo = lexical_score(q, "The meeting is on Tuesday")
        self.assertGreater(hi, lo)
        self.assertEqual(lexical_score(q, "zzzz unrelated"), 0.0)


class ParseFactsTest(unittest.TestCase):
    def test_plain_array(self):
        raw = '[{"text": "User name is Sam", "kind": "person"}]'
        facts = parse_fact_payload(raw)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["kind"], "person")

    def test_fenced_and_empty(self):
        raw = "```json\n[]\n```"
        self.assertEqual(parse_fact_payload(raw), [])
        self.assertEqual(parse_fact_payload("no json here"), [])

    def test_string_items(self):
        facts = parse_fact_payload('["User lives in Bangalore"]')
        self.assertEqual(facts[0]["text"], "User lives in Bangalore")


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "mem"
        notes = Path(self.tmp.name) / "notes"
        notes.mkdir()
        self.store = MemoryStore(str(root), str(notes), keep_turns=2, fact_limit=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_and_dedup(self):
        n = self.store.add_facts([
            {"text": "User prefers Jarvis voice", "kind": "preference"},
            {"text": "User prefers Jarvis voice", "kind": "preference"},
        ])
        self.assertEqual(n, 1)
        self.assertEqual(len(self.store._facts), 1)

    def test_retrieve_prefers_overlap_else_recent(self):
        self.store.add_facts([
            {"text": "User drinks espresso", "kind": "preference"},
            {"text": "Project Vox is a voice assistant", "kind": "project"},
            {"text": "Dash 4 is the speaker", "kind": "other"},
        ])
        hits = self.store.retrieve_facts("what speaker do I use")
        self.assertTrue(any("Dash 4" in h["text"] for h in hits))

        none_match = self.store.retrieve_facts("zzzzzyx")
        self.assertEqual(none_match[0]["text"], "Dash 4 is the speaker")

    def test_context_includes_facts(self):
        self.store.add_facts([{"text": "User name is Sam", "kind": "person"}])
        block = self.store.context_for("what is my name")
        self.assertIn("User name is Sam", block)
        self.assertIn("## Memory", block)

    def test_notes_search_on_recall(self):
        md = self.store.notes_dir / "session.md"
        md.write_text("# Session\nWe decided to use Airtel Batcave for downloads.\n")
        block = self.store.context_for("remember what we decided about Batcave")
        self.assertIn("Past notes", block)
        self.assertIn("Batcave", block)

    def test_forget(self):
        self.store.add_facts([{"text": "A", "kind": "other"}, {"text": "B", "kind": "other"}])
        self.assertEqual(self.store.forget_last(), "B")
        self.store.forget_all()
        self.assertEqual(self.store._facts, [])

    def test_observe_stores_parsed_facts(self):
        llm = FakeLLM('[{"text": "User likes concise replies", "kind": "preference"}]')
        self.store.observe(llm, "keep it short", "Sure.")
        self.assertEqual(self.store._facts[0]["text"], "User likes concise replies")

    def test_compact_trims_history(self):
        llm = FakeLLM("User discussed speakers and then coffee.")
        history = []
        for i in range(6):
            history.append({"role": "user", "content": f"u{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        trimmed = self.store.compact(llm, history)
        self.assertEqual(len(trimmed), 4)  # keep_turns=2 → 4 messages
        self.assertIn("coffee", self.store.session_summary)

    def test_forget_phrases(self):
        self.assertTrue(self.store.is_forget_all("forget everything"))
        self.assertTrue(self.store.is_forget_last("forget that."))
        self.assertFalse(self.store.is_forget_all("forget the milk"))

    def test_facts_persist_to_disk(self):
        self.store.add_facts([{"text": "Persisted fact", "kind": "other"}])
        reloaded = MemoryStore(str(self.store.root), str(self.store.notes_dir))
        self.assertEqual(reloaded._facts[0]["text"], "Persisted fact")


if __name__ == "__main__":
    unittest.main()
