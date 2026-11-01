"""The brain plans a tool turn while generating and applies it only at commit."""

import re
import threading
import unittest

from agent_tools import Plan
from test_vself_brain import FakeRoute, brain


class FakeTools:
    def __init__(self, plans=None, notice=None):
        self.plans = plans or {}
        self.notice = notice
        self.applied = []
        self.done = threading.Event()

    def capabilities_block(self):
        return ""

    def call_agent(self, agent, question, by=""):
        self.calls = getattr(self, "calls", []) + [(agent, question)]
        self.by = by
        return Plan(f"{agent} says: 4 gigabytes.", "ask")

    def plan(self, text, resolve=True):
        p = self.plans.get(text)
        if p:
            return Plan(p, "notify", lambda: (self.applied.append(text), self.done.set()))
        return None

    def finish(self, plan):
        return plan.lazy() if plan.lazy else plan

    def take_notice(self):
        n, self.notice = self.notice, None
        return n


def make(tools):
    local = FakeRoute("local", ["model ", "answer."])
    b = brain(local, None)
    b.tools = tools
    return b, local


class BrainToolsTest(unittest.TestCase):
    def test_tool_turn_speaks_the_plan_and_skips_the_model_and_vself_prepare(self):
        b, local = make(FakeTools({"notify me hi": "Sending that."}))
        out = "".join(b.chat_stream("notify me hi", commit=False))
        self.assertEqual(out, "Sending that.")
        self.assertEqual(local.calls, 0)
        self.assertFalse([p for p in b.api.posts if p[0].endswith("prepare")])
        self.assertEqual(b.last_route, "tool:notify")

    def test_nothing_is_applied_until_commit(self):
        t = FakeTools({"notify me hi": "Sending that."})
        b, _ = make(t)
        "".join(b.chat_stream("notify me hi", commit=False))
        self.assertEqual(t.applied, [])
        b.commit("notify me hi", "Sending that.")
        self.assertTrue(t.done.wait(2))
        self.assertEqual(t.applied, ["notify me hi"])

    def test_a_discarded_speculative_turn_never_applies(self):
        t = FakeTools({"notify me hi": "Sending that."})
        b, _ = make(t)
        "".join(b.chat_stream("notify me hi", commit=False))  # the user kept talking; this turn is dropped
        "".join(b.chat_stream("never mind, what is the weather", commit=False))
        b.commit("never mind, what is the weather", "It is sunny.")
        self.assertFalse(t.done.wait(0.3))
        self.assertEqual(t.applied, [])

    def test_commit_applies_once(self):
        t = FakeTools({"notify me hi": "Sending that."})
        b, _ = make(t)
        "".join(b.chat_stream("notify me hi", commit=False))
        b.commit("notify me hi", "Sending that.")
        b.commit("notify me hi", "Sending that.")
        t.done.wait(2)
        self.assertEqual(t.applied, ["notify me hi"])

    def test_a_long_tool_answer_is_trimmed_in_history_but_chat_turns_are_kept_whole(self):
        long = "DUM-E says: " + "the main disk is ninety one percent full and getting high. " * 8
        b, _ = make(FakeTools({"ask dume status": long}))
        "".join(b.chat_stream("ask dume status", commit=False))
        b.commit("ask dume status", long)
        b.commit("hello there", long)  # a normal chat turn is never trimmed
        self.assertLessEqual(len(b.history[-3]["content"]), 165)
        self.assertEqual(b.history[-1]["content"], long)

    def test_ordinary_talk_still_goes_to_the_model(self):
        b, local = make(FakeTools({"notify me hi": "x"}))
        out = "".join(b.chat_stream("how are you", commit=False))
        self.assertEqual(out, "model answer.")
        self.assertEqual(local.calls, 1)

    def test_a_failed_earlier_action_is_mentioned_before_the_next_answer(self):
        b, _ = make(FakeTools(notice="The last thing I tried didn't work."))
        out = "".join(b.chat_stream("how are you", commit=False))
        self.assertTrue(out.startswith("The last thing I tried didn't work. "))
        self.assertTrue(out.endswith("model answer."))

    def test_a_slow_agent_gets_an_immediate_ack_before_its_answer(self):
        order = []
        t = FakeTools()
        t.plan = lambda text, resolve=True: Plan("", "ask", ack="Asking Dexter.",
                                                 lazy=lambda: (order.append("answer computed"), Plan("Dexter says: up two percent.", "ask"))[1])
        b, _ = make(t)
        pieces = []
        for piece in b.chat_stream("how is nvidia stock", commit=False):
            pieces.append(piece)
            order.append("spoke " + piece.strip())
        self.assertEqual(order[0], "spoke Asking Dexter.")  # the voice has the ack before the slow call even starts
        self.assertLess(order.index("spoke Asking Dexter."), order.index("answer computed"))
        self.assertEqual("".join(pieces), "Asking Dexter. Dexter says: up two percent.")

    def test_long_tool_speech_is_yielded_in_short_clauses_that_join_back_unchanged(self):
        text = "The machine is up, but the main disk is 91 percent full. The load is light, the AI brain is running."
        b, _ = make(FakeTools({"status": text}))
        pieces = list(b.chat_stream("status", commit=False))
        self.assertGreater(len(pieces), 3)
        self.assertTrue(all(len(p.split()) <= 12 for p in pieces))
        self.assertEqual("".join(pieces), text)

    def test_a_brain_without_tools_behaves_as_before(self):
        b, local = make(None)
        self.assertEqual("".join(b.chat_stream("notify me hi", commit=False)), "model answer.")


class ModelCallsAnAgentTest(unittest.TestCase):
    """The language model can ask an agent itself, so the owner's wording never has to match a phrase."""

    def run_model(self, tokens, tools=None):
        local = FakeRoute("local", tokens)
        b = brain(local, None)
        b.tools = tools or FakeTools()
        out = "".join(b.chat_stream("how much ram is in use", commit=False))
        return b, out

    def test_a_call_line_runs_the_agent_and_only_its_answer_is_spoken(self):
        tools = FakeTools()
        b, out = self.run_model(["[[ask:dume|how much ", "RAM is in use]]", " (and then some chatter)"], tools)
        self.assertEqual(tools.calls, [("dume", "how much RAM is in use")])
        self.assertEqual(out, "dume says: 4 gigabytes.")
        self.assertNotIn("[[", out)
        self.assertEqual(b.last_route, "tool:ask")

    def test_a_call_line_split_across_many_tokens_still_works(self):
        tools = FakeTools()
        _, out = self.run_model(list("[[ask:dume|ram?]]"), tools)
        self.assertEqual(tools.calls, [("dume", "ram?")])
        self.assertEqual(out, "dume says: 4 gigabytes.")

    def test_normal_replies_are_not_delayed_or_changed(self):
        tools = FakeTools()
        _, out = self.run_model(["Sure, ", "here you go."], tools)
        self.assertEqual(out, "Sure, here you go.")
        self.assertFalse(hasattr(tools, "calls"))

    def test_a_reply_that_merely_starts_with_a_bracket_is_spoken(self):
        tools = FakeTools()
        _, out = self.run_model(["[laughs] ", "good one."], tools)
        self.assertEqual(out, "[laughs] good one.")
        self.assertFalse(hasattr(tools, "calls"))

    def test_an_unknown_agent_or_a_flowsmith_line_is_never_run(self):
        for line in ["[[ask:root|rm -rf /]]", "[[ask:flowsmith|build me a workflow]]", "[[ask:dume|]]"]:
            tools = FakeTools()
            _, out = self.run_model([line], tools)
            self.assertFalse(hasattr(tools, "calls"), line)

    def test_a_call_with_no_tools_attached_is_spoken_as_text(self):
        b = brain(FakeRoute("local", ["[[ask:dume|ram]]"]), None)
        b.tools = None
        self.assertEqual("".join(b.chat_stream("hi", commit=False)), "[[ask:dume|ram]]")


class CallLineAnywhereTest(unittest.TestCase):
    def run_model(self, tokens):
        tools = FakeTools()
        b = brain(FakeRoute("local", tokens), None)
        b.tools = tools
        return tools, "".join(b.chat_stream("is my graphics card okay", commit=False))

    def test_a_hedging_sentence_before_the_call_is_dropped_with_it(self):
        reply = "I can check that for you right now. [[ask:dume|what model is on the GPU and any errors]]"
        for tokens in ([reply], list(reply), re.findall(r"\S+\s*", reply)):
            tools, out = self.run_model(tokens)
            self.assertEqual(tools.calls, [("dume", "what model is on the GPU and any errors")])
            self.assertEqual(out, "dume says: 4 gigabytes.")

    def test_a_normal_sentence_before_the_call_is_spoken_but_the_call_line_never_is(self):
        reply = "Sure, give me a second. [[ask:dume|how is the gpu]] That is all."
        for tokens in ([reply], list(reply), re.findall(r"\S+\s*", reply)):
            tools, out = self.run_model(tokens)
            self.assertEqual(tools.calls, [("dume", "how is the gpu")])
            self.assertNotIn("[[", out)
            self.assertNotIn("That is all", out)
            self.assertTrue(out.startswith("Sure, give me a second. "))

    def test_brackets_that_are_not_a_call_are_spoken_whole(self):
        for reply in ["Sure thing, [[this is not a call]] anyway.", "I like [a] lot of [b] things, honestly.",
                      "Sure, [[ask:root|x]] is not allowed."]:
            for tokens in ([reply], list(reply)):
                tools, out = self.run_model(tokens)
                self.assertFalse(hasattr(tools, "calls"), reply)
                self.assertEqual(out, reply)


class PromiseToAskTest(unittest.TestCase):
    """A reply that says it will ask an agent (or tells the owner to) is a call that was never made."""

    def run_model(self, tokens):
        tools = FakeTools()
        b = brain(FakeRoute("local", tokens), None)
        b.tools = tools
        out = "".join(b.chat_stream("is my graphics card okay", commit=False))
        return tools, out

    def test_the_replies_the_model_really_gave_become_real_calls(self):
        for reply in ["I don't have that data right now. Ask DUM-E how much RAM is being used.",
                      "I can\u2019t check the card myself. Ask DUM-E to run nvidia-smi for you and tell me what it shows.",
                      "I will ask DUM-E for the current GPU memory and temperature now.",
                      "I can ask DUM-E to check the GPU memory and temperature right now.",
                      "Ask DUM-E to check the memory usage for you.",
                      "You can ask Dexter about that."]:
            for tokens in ([reply], list(reply), re.findall(r"\S+\s*", reply)):  # one chunk, per character, per word
                tools, out = self.run_model(tokens)
                self.assertEqual(len(getattr(tools, "calls", [])), 1, reply)
                self.assertEqual(tools.calls[0][1], "is my graphics card okay")
                self.assertEqual(out, f"{tools.calls[0][0]} says: 4 gigabytes.", reply)

    def test_ordinary_hedges_and_denials_are_spoken_in_full(self):
        for reply in ["I can't tell you what the weather is like. Sorry about that.",
                      "I think the card is fine, honestly. It has been quiet.",
                      "Sorry, I missed that. Say it again?",
                      "Sure, here you go."]:
            tools, out = self.run_model(list(reply))
            self.assertFalse(hasattr(tools, "calls"), reply)
            self.assertEqual(out, reply)

    def test_normal_replies_start_speaking_without_waiting_for_the_end(self):
        local = FakeRoute("local", ["Honestly, ", "the card is fine. ", "Nothing odd. ", "All quiet."])
        b = brain(local, None)
        b.tools = FakeTools()
        first = next(iter(b.chat_stream("is my graphics card okay", commit=False)))
        self.assertTrue(first.startswith("Honestly"))  # decided on the first words, not the whole reply


if __name__ == "__main__":
    unittest.main()
