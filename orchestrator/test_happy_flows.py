"""The owner's own words. Every line here was really said to Apollo (from the saved transcripts), so a
phrasing I did not think of can no longer slip through. Add a line whenever something real is missed."""
import unittest

import agent_tools
from test_agent_tools import ToolCase

# said, and an agent (or the roster) should take it
ROUTED_TO_DUME = [
    "Can you ask dummy what is going on in the system?",
    "Can you ask Diom what is the status of the system? Everything holding up.",
    "What is the status of the machine right now?",
    "What is the status of the machine?",
    "Ask dummy about the machine status.",
    "Kenya's dummy, what are the models loaded in the GPU?",
    "ask dummy what are the models loaded in the gpu",
    "No, I mean can you tell me the exact model which is running right now in the GPU?",
    "Can you ask dummy to list the specific model names?",
    "Anyways tell me what's the exact status of the hard disk right now across the machine",
    "What is the status of all the storage devices on the machine?",
    "Can you tell how much RAM is being used right now?",
    "Can you tell me how much RAM is being used right now?",
    "how much memory is being used",
    "is the gpu busy",
    "how much disk space is left",
    "which containers are running",
]

# said, and it is just conversation (an agent must not be started, nothing on the server touched)
CONVERSATION = [
    "Did you actually ask dummy?",
    "Did you just call dummy for that? I don't think you did",
    "don't you think it should be called dumb e that's why it is",
    "Now I mean to ask you what do you think of billionaires controlling us, the majority of the people?",
    "Yeah, if you're not gonna talk what I am asking you to talk then what's the whole point of it?",
    "I simply ask you to repeat something. This is not a threat or anything. This is just for fun for me.",
    "I'm facing a conundrum. I don't want to use a cloud-based AI model.",
    "But my local tech stack is not powerful enough to host a great AI model which is like",
    "Do you have access to FlowSmith?",
    "you know i have so much to do right now i mean i can do right now i don't have to do anything",
    "I mean, I'm learning music theory. I have to work on Apache Data Fusion.",
    "I have to research on a story for an episodic season or a short film.",
    "what do you think about the system of government",
    "I have a good memory of that trip",
    "my computer science exam is tomorrow",
    "I have sixteen gigs of VRAM and I keep running out when two models load. What would you do?",
    "how should I set up my disks for a new build",
    "what do you recommend for a gpu under a thousand dollars",
    "Tell me the summary of the movie You Can See Everything that has been released this year.",
    "what is the plot of the new Dune movie",
    "who won the cricket match yesterday",
]


class RealUtterancesTest(ToolCase):
    def test_every_real_machine_question_reaches_dume(self):
        for q in ROUTED_TO_DUME:
            self.world.calls.clear()
            plan = self.say(q)
            self.assertIsNotNone(plan, f"not routed: {q}")
            self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1, f"did not reach DUM-E: {q}")
            self.assertTrue(plan.speak.startswith("DUM-E"), f"{q!r} -> {plan.speak!r}")

    def test_real_conversation_is_left_alone(self):
        for q in CONVERSATION:
            self.world.calls.clear()
            self.assertIsNone(self.tools.plan(q), f"hijacked: {q}")
            self.assertEqual(self.world.calls, [], q)

    def test_the_agents_question_is_answered_with_the_roster(self):
        for q in ["what all agents you have access to.", "Are you aware which AI agent handles all the things in the system?"]:
            self.assertIn("I'm the hub", self.say(q).speak, q)

    def test_ask_it_for_me_replays_what_was_just_asked_however_it_is_put(self):
        for follow in ["Can you ask it for me?", "I mean you can ask it for me", "just ask it for me", "you can ask it"]:
            self.world.calls.clear()
            self.assertIsNone(self.tools.plan("what's the weather like"))  # not a tool request; remembered as the last thing said
            self.tools.plan("Can you tell how much RAM is being used right now?")
            self.world.calls.clear()
            plan = self.say(follow)
            self.assertIsNotNone(plan, follow)
            self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1, follow)

    def test_yes_after_a_real_question_is_not_a_tool_call(self):
        self.say("Can you tell how much RAM is being used right now?")
        self.world.calls.clear()
        self.assertIsNone(self.tools.plan("Yes."))
        self.assertEqual(self.world.calls, [])

    def test_the_model_is_told_how_to_call_agents_and_never_to_deny_them(self):
        block = self.tools.capabilities_block()
        self.assertIn("[[ask:AGENT|", block)
        self.assertIn("Never say you cannot reach an agent", block)
        self.assertNotIn("You cannot start anything between turns", block)

    def test_the_model_is_told_detective_browses_and_cites_sources(self):
        block = self.tools.capabilities_block()
        self.assertIn("-> detective, which browses and cites sources", block)
        self.assertIn("15 to 40 seconds", block)
        self.assertIn("chit-chat, opinions or things you already know for certain", block)
        self.assertNotIn("it cannot browse", block)
        self.assertNotIn("Never send films", block)
        self.assertIn("browses the web through its own scouts", agent_tools.BLURBS["detective"])

    def test_the_audit_keeps_the_agents_full_reply_and_who_chose_the_call(self):
        self.world.chat_reply = {"reply": "Outside my beat."}
        plan = self.tools.call_agent("detective", "plot of a film", by="cursor:composer-2.5-fast")
        self.tools.finish(plan)
        row = self.audit()[-1]
        self.assertEqual((row["agent"], row["by"], row["reply"]), ("detective", "cursor:composer-2.5-fast", "Outside my beat."))

    def test_a_call_made_by_the_phrase_rules_says_so(self):
        self.say("ask the detective about the budget")
        self.assertEqual(self.audit()[-1]["by"], "rules")

    def test_call_line_parsing(self):
        P = agent_tools.parse_call
        self.assertEqual(P("[[ask:dume|how much RAM is in use]]"), ("dume", "how much RAM is in use"))
        self.assertEqual(P("  [[ ASK : Detective | latest on the port strike ]] trailing"), ("detective", "latest on the port strike"))
        for bad in ["[[ask:flowsmith|x]]", "[[ask:root|x]]", "[[ask:dume|]]", "ask dume", "[laughs]", "[[ask:dume|" + "x" * 500 + "]]"]:
            self.assertIsNone(P(bad), bad)


if __name__ == "__main__":
    unittest.main()
