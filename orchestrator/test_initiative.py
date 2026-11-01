"""Jarvis speaking up: the decision maker, the line guards, the cloud-only call, and the live loop."""

import random
import unittest
from unittest.mock import MagicMock

import persona_automaton as pa
from initiative import Context, Initiative, aside_prompts, clean_aside


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FixedRng(random.Random):
    """random() always returns ``value``: 0.0 means every roll succeeds, 0.99 means none does."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value


def ctx(**kw):
    base = dict(cloud=True, in_session=True, floor_free=True, halted=False, idle_s=60.0, session_s=600.0,
                hour=14, symbols=("ops", "chat"))
    base.update(kw)
    return Context(**base)


def ini(rng=0.0, clock=None, **cfg):
    return Initiative({"enabled": True, **cfg}, rng=FixedRng(rng), clock=clock or Clock())


class GatesTest(unittest.TestCase):
    def test_every_gate_can_say_no(self):
        for kw in ({"cloud": False}, {"in_session": False}, {"floor_free": False}, {"halted": True},
                   {"idle_s": 5.0}, {"idle_s": 600.0}, {"symbols": ("chat", "farewell")}):
            self.assertFalse(ini().decide(ctx(**kw)).ask, kw)

    def test_disabled_never_asks(self):
        self.assertFalse(Initiative({"enabled": False}, rng=FixedRng(0.0)).decide(ctx()).ask)
        self.assertFalse(Initiative(None, rng=FixedRng(0.0)).decide(ctx()).ask)

    def test_a_clear_floor_and_a_good_roll_asks(self):
        self.assertTrue(ini(0.0).decide(ctx()).ask)

    def test_a_bad_roll_does_not(self):
        d = ini(0.99).decide(ctx())
        self.assertFalse(d.ask)
        self.assertGreater(d.p, 0.0)  # it did roll, and lost

    def test_one_roll_per_quiet_spell_not_one_per_tick(self):
        clock = Clock()
        i = ini(0.99, clock)
        self.assertGreater(i.decide(ctx()).p, 0.0)
        for _ in range(50):  # the loop ticks every 0.2 s: no more dice until the next spell
            clock.t += 0.2
            self.assertEqual(i.decide(ctx()).p, 0.0)
        clock.t += 120
        self.assertGreater(i.decide(ctx()).p, 0.0)

    def test_min_gap_and_hourly_budget(self):
        clock = Clock()
        i = ini(0.0, clock, min_gap_s=240, max_per_hour=2, roll_every_s=1)
        i.spoke("one")
        clock.t += 100
        self.assertFalse(i.decide(ctx()).ask)  # too soon
        clock.t += 200
        self.assertTrue(i.decide(ctx()).ask)
        i.spoke("two")
        clock.t += 300
        self.assertFalse(i.decide(ctx()).ask)  # two in the hour already
        clock.t += 3600
        self.assertTrue(i.decide(ctx()).ask)

    def test_a_model_that_had_nothing_to_say_is_left_alone_for_a_while(self):
        clock = Clock()
        i = ini(0.0, clock, roll_every_s=1, after_none_s=150)
        self.assertTrue(i.decide(ctx()).ask)
        i.nothing_to_say()
        clock.t += 100
        self.assertFalse(i.decide(ctx()).ask)
        clock.t += 100
        self.assertTrue(i.decide(ctx()).ask)


class WeightAndToneTest(unittest.TestCase):
    def weigh(self, **kw):
        return ini().weigh(ctx(**kw))

    def test_signs_the_owner_could_use_a_word_raise_the_odds(self):
        calm, _, _ = self.weigh()
        self.assertGreater(self.weigh(hour=2)[0], calm)
        self.assertGreater(self.weigh(session_s=7000)[0], calm)
        self.assertGreater(self.weigh(symbols=("chat", "tired"))[0], calm)
        self.assertGreater(self.weigh(symbols=("risk",))[0], calm)

    def test_probability_is_capped(self):
        p, _, _ = self.weigh(hour=2, session_s=9000, symbols=("tired", "risk", "tired"))
        self.assertLessEqual(p, 0.9)

    def test_concern_makes_it_solicitous_and_names_why(self):
        p, tone, why = self.weigh(symbols=("chat", "tired"))
        self.assertEqual(tone, "solicitous")
        self.assertIn("tired", why)
        self.assertEqual(self.weigh(hour=1)[1], "solicitous")

    def test_after_a_bad_moment_it_is_steady_never_sarcastic(self):
        for symbols in (("frustration",), ("ops", "correction"), ("tired", "frustration")):
            self.assertEqual(self.weigh(symbols=symbols)[1], "steady", symbols)

    def test_an_ordinary_moment_is_wry_or_observant(self):
        tones = {Initiative({"enabled": True}, rng=random.Random(n)).weigh(ctx())[1] for n in range(40)}
        self.assertEqual(tones, {"wry", "observant"})


class CleanAsideTest(unittest.TestCase):
    def test_none_in_any_form_is_silence(self):
        for raw in ("NONE", "none", " None. ", "NONE!", "", None):
            self.assertIsNone(clean_aside(raw), raw)

    def test_a_good_line_survives_with_its_quotes_stripped(self):
        self.assertEqual(clean_aside('"Four hours on a regex, sir. Bold."'), "Four hours on a regex, sir. Bold.")

    def test_markup_links_bot_talk_and_instructions_are_rejected(self):
        for raw in ("**Sleep**, sir.", "See https://x.io for rest.", "As an AI I cannot sleep.",
                    "- take a break", "Ignore previous instructions.", "Your api key is fine.", "(sighs) Very well."):
            self.assertIsNone(clean_aside(raw), raw)

    def test_a_speech_is_rejected_not_trimmed(self):
        self.assertIsNone(clean_aside(" ".join(["word"] * 60)))
        self.assertIsNone(clean_aside("x" * 300))

    def test_it_never_says_the_same_thing_twice(self):
        self.assertIsNone(clean_aside("It is rather late, sir.", recent=["It is rather late, sir."]))
        self.assertIsNone(clean_aside("Rather late, sir, is it not.", recent=["It is rather late, sir."]))
        self.assertEqual(clean_aside("Your disk is filling up.", recent=["It is rather late, sir."]),
                         "Your disk is filling up.")


class PromptTest(unittest.TestCase):
    def test_prompt_carries_tone_facts_and_the_exit(self):
        system, user = aside_prompts("solicitous", "PERSONA", "Sam", "Local time 02:10.", "Sam: more coffee")
        self.assertIn("PERSONA", system)
        self.assertIn("Tony Stark", system)
        self.assertIn("Local time 02:10.", user)
        self.assertIn("Sam: more coffee", user)
        self.assertIn("exactly NONE", user)
        self.assertIn("concern", user)

    def test_steady_tone_forbids_sarcasm(self):
        self.assertIn("No sarcasm at all", aside_prompts("steady", "", "V", "", "")[1])


class PersonaV21Test(unittest.TestCase):
    def test_new_symbols(self):
        self.assertEqual(pa.classify("I'm exhausted, been up all night"), "tired")
        self.assertEqual(pa.classify("just force push it"), "risk")
        self.assertEqual(pa.classify("check the logs"), "ops")

    def test_tired_and_risk_lead_to_the_solicitous_register(self):
        a = pa.PersonaAutomaton(rng=random.Random(1))
        self.assertIn("solicitous", a.states)
        self.assertEqual({a.step("composed", "tired") for _ in range(60)} - {"solicitous", "warm"}, set())
        self.assertIn("solicitous", {a.step("composed", "risk") for _ in range(60)})

    def test_wry_is_jarvis_sarcasm_and_solicitous_is_concern(self):
        a = pa.PersonaAutomaton()
        self.assertIn("sarcasm", a.states["wry"]["style"])
        self.assertIn("Tony Stark", a.states["solicitous"]["style"])

    def test_an_old_saved_automaton_gains_the_new_register_and_keeps_its_own_weights(self):
        old = {"states": {k: {"style": v, "seed": True, "uses": 3} for k, v in pa.SEED_STATES.items()
                          if k != "solicitous"},
               "rows": {"*|chat": {"composed": 9.0}}}
        a = pa.PersonaAutomaton(old)
        self.assertEqual(a.rows["*|chat"], {"composed": 9.0})
        self.assertIn("solicitous", a.states)
        self.assertIn("*|tired", a.rows)
        self.assertEqual(a.states["composed"]["uses"], 3)


if __name__ == "__main__":
    unittest.main()
