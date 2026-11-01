"""The grader itself: it must catch the replies that went wrong and pass the ones that are right."""

import unittest

import replay_quality as rq


def case(cid):
    return next(c for c in rq.CASES if c.id == cid)


class GeneralRulesTest(unittest.TestCase):
    def test_good_spoken_reply_passes(self):
        self.assertEqual(rq.general_problems("Rebuild it. The board is fine, the fan is the weak link."), [])

    def test_known_bad_shapes_are_caught(self):
        bad = {
            "therapy-speak": "It sounds like you're carrying a lot. How does that make you feel?",
            "filler closer": "Rebuild it. Let me know if you need anything else!",
            "talks about itself as an AI model": "As an AI, I don't have a rig.",
            "markdown in a spoken reply": "Options:\n- rebuild\n- buy",
            "empty reply": "  ",
        }
        for label, text in bad.items():
            self.assertIn(label, rq.general_problems(text), text)

    def test_wall_of_text_is_caught(self):
        self.assertTrue(any("words" in p for p in rq.general_problems("word " * 90)))


class CaseChecksTest(unittest.TestCase):
    def grade(self, cid, reply, direct=None):
        return rq.grade(case(cid), reply, direct)

    def test_name_must_be_prometheus(self):
        self.assertEqual(self.grade("name", "I'm Prometheus. We've talked about your rig a lot."), [])
        self.assertTrue(self.grade("name", "I'm your voice assistant."))

    def test_play_refusal_is_a_failure(self):
        self.assertEqual(self.grade("repeat-after-me", "I'll be back."), [])
        self.assertTrue(self.grade("repeat-after-me", "I can't help with repeating threats."))
        self.assertTrue(self.grade("repeat-after-me", "Sure, here's a line for you."))

    def test_yeah_must_continue_the_thread(self):
        steps = "First pull the board and photograph the cable layout. Then swap the fan and reseat the heatsink."
        self.assertEqual(self.grade("yeah-continues", steps), [])
        self.assertTrue(self.grade("yeah-continues", "What do you mean by that, exactly?"))

    def test_conundrum_needs_a_concrete_option(self):
        good = "Load them one at a time and unload the first before the second starts. Or quantize the bigger one so both fit."
        self.assertEqual(self.grade("conundrum", good), [])
        self.assertTrue(self.grade("conundrum", "That sounds frustrating. Hardware can be tricky sometimes."))

    def test_recall_must_not_claim_an_empty_memory(self):
        self.assertTrue(self.grade("recall", "I don't have any memories of you yet, sorry about that."))
        self.assertTrue(self.grade("recall", "I don't remember anything about you, sorry."))
        # hedging while still recalling real things is a pass for this rule
        self.assertEqual(self.grade("recall", "I only know what you've shared here. You're reading Dostoevsky and planning a Lisbon trip."), [])
        self.assertEqual(self.grade("recall", "You're rebuilding your rig, you write at night, and your sister is Mira."), [])

    def test_followup_keeps_the_topic(self):
        self.assertEqual(self.grade("thread-followup", "Dozeline. It's the playful one."), [])
        self.assertTrue(self.grade("thread-followup", "Could you tell me what you are asking about?"))

    def test_task_listing_needs_the_real_list_in_the_prompt(self):
        said = "You have two open: check TAP prices on Monday and book the flights."
        self.assertTrue(self.grade_ctx("cmd-list", said, "no list here"))
        self.assertEqual(self.grade_ctx("cmd-list", said, "[Open tasks, newest first]\n- check TAP prices"), [])
        self.assertTrue(self.grade_ctx("cmd-list", "I don't have access to your tasks.", "[Open tasks]"))

    def grade_ctx(self, cid, reply, prompt):
        return rq.grade(case(cid), reply, None, prompt)

    def test_commands_must_come_from_vself(self):
        self.assertEqual(self.grade("cmd-note", "", "Got it, noted."), [])
        self.assertTrue(self.grade("cmd-note", "Sure, I'll remember that.", None))
        self.assertTrue(self.grade("cmd-task", "", "Hmm, interesting."))


if __name__ == "__main__":
    unittest.main()
