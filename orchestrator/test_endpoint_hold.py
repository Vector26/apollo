import unittest

from endpoint_hold import FragmentHold, ends_dangling, is_filler_only


class FillerTests(unittest.TestCase):
    def test_filler_only_turns_are_nothing(self):
        for text in ("Um...", "Uh, hmm.", "hmm", "Um, uh."):
            self.assertTrue(is_filler_only(text), text)

    def test_real_words_are_not_filler(self):
        for text in ("Um, okay", "Yeah.", "Hmm, that's interesting", ""):
            self.assertFalse(is_filler_only(text), text)


class DanglingTests(unittest.TestCase):
    def test_unfinished_thoughts_are_held(self):
        for text in ("I just", "And then I wanted to", "so, um,", "It was the", "I think but"):
            self.assertTrue(ends_dangling(text), text)

    def test_finished_thoughts_are_answered(self):
        for text in (
            "What do you think about that?",
            "Tell me about my day.",
            "I just got back",
            "How are you",
            "Why did that happen?",
            "Thank you!",
        ):
            self.assertFalse(ends_dangling(text), text)

    def test_questions_never_dangle(self):
        self.assertFalse(ends_dangling("Is it a or b?"))

    def test_filler_is_dropped_not_held(self):
        self.assertFalse(ends_dangling("Um..."))


class HoldTests(unittest.TestCase):
    def test_continuation_is_merged_without_double_punctuation(self):
        h = FragmentHold()
        h.hold("I just...")
        self.assertEqual(h.peek_merge("wanted to ask something"), "I just wanted to ask something")

    def test_peek_is_repeatable_until_cleared(self):
        h = FragmentHold()
        h.hold("And then")
        self.assertEqual(h.peek_merge("it broke"), h.peek_merge("it broke"))
        h.clear()
        self.assertEqual(h.peek_merge("it broke"), "it broke")

    def test_expires_after_the_window(self):
        now = [100.0]
        h = FragmentHold(window_s=2.5, clock=lambda: now[0])
        h.hold("I just")
        self.assertFalse(h.expired())
        now[0] += 2.6
        self.assertTrue(h.expired())

    def test_sequence_changes_with_each_hold(self):
        h = FragmentHold()
        first = h.hold("a")
        self.assertNotEqual(first, h.hold("b"))

    def test_take_returns_and_empties(self):
        h = FragmentHold()
        h.hold("so")
        self.assertEqual(h.take(), "so")
        self.assertFalse(h.pending)


if __name__ == "__main__":
    unittest.main()
