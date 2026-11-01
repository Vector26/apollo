import unittest

from carry import MAX_AGE_S, UnansweredCarry


class CarryTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.c = UnansweredCarry(clock=lambda: self.now)

    def test_nothing_carried_means_the_text_is_untouched(self):
        self.assertEqual(self.c.merged("hello"), "hello")

    def test_unanswered_utterances_come_first_in_order(self):
        self.c.add("I have so much to do.")
        self.c.add("I'm learning music theory.")
        self.assertEqual(self.c.merged("I have to research a story."),
                         "I have so much to do. I'm learning music theory. I have to research a story.")

    def test_settling_drops_only_what_was_answered(self):
        self.c.add("one")
        taken = self.c.pending()
        self.c.add("two")  # arrived after the turn that carried "one" began
        self.c.settle(taken)
        self.assertEqual(self.c.pending(), ["two"])

    def test_a_turn_that_is_cancelled_again_keeps_everything_for_the_next(self):
        self.c.add("one")
        self.c.add("two")  # the merged turn was barged in too: its own raw text is added, nothing settled
        self.assertEqual(self.c.merged("three"), "one two three")

    def test_old_and_blank_items_are_dropped(self):
        self.c.add("   ")
        self.c.add("stale")
        self.now += MAX_AGE_S + 1
        self.c.add("fresh")
        self.assertEqual(self.c.pending(), ["fresh"])

    def test_at_most_four_are_kept(self):
        for i in range(7):
            self.c.add(f"s{i}")
        self.assertEqual(self.c.pending(), ["s3", "s4", "s5", "s6"])


if __name__ == "__main__":
    unittest.main()
