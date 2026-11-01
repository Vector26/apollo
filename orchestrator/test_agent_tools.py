"""Apollo's tools: routing, confirmation, scoping, and failure wording. Gateway and notifier are faked."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import agent_tools
from agent_tools import AgentTools, speakable

REGISTRY = """
version: 5
agents:
  - {id: detective, name: The Detective, status: active, api: {adapter: detective, enabled: true, risk_class: intel,
      allowed_job_types: [run, resume, nightly, status], default_job_type: run, read_only_job_types: [status]}}
  - {id: dexter, name: Dexter, status: active, api: {adapter: dexter, enabled: true, risk_class: finance, default_job_type: research}}
  - {id: oda, name: ODA, status: active, api: {adapter: oda, enabled: true, risk_class: design,
      default_job_type: status, read_only_job_types: [status, previews]}}
  - {id: dume, name: DUM-E, status: active, api: {adapter: dume, enabled: true, risk_class: ops, default_job_type: snapshot,
      read_only_job_types: [snapshot]}}
  - {id: flowsmith, name: flowsmith, status: active, experimental: true, api: {adapter: flowsmith, enabled: true,
      risk_class: automation, default_job_type: task, read_only_job_types: [list_workflows]}}
"""


class FakeWorld:
    """Records every call; routes are set per test."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None, dict]] = []
        self.chat_reply = {"reply": "All quiet."}
        self.status = 200
        self.job_rows: dict[str, dict] = {}
        self.fail_connect = False
        self.notifier_status = 200
        self.chat_status = 200
        self.chat_delay = 0.0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail_connect:
            raise httpx.ConnectError("down")
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body, dict(request.headers)))
        path = request.url.path
        if path == "/notify":
            return httpx.Response(self.notifier_status, json={})
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": self.detail}, headers={"Retry-After": "60"})
        if path.endswith("/chat"):
            if self.chat_delay:
                time.sleep(self.chat_delay)
            if self.chat_status != 200:
                return httpx.Response(self.chat_status, json={"detail": "boom"})
            return httpx.Response(200, json=self.chat_reply)
        if path.endswith("/health"):
            return httpx.Response(200, json={"breaker": "closed", "running": 0, "queued": 0})
        if path.endswith("/jobs") and request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1", "status": "queued"})
        if path.startswith("/jobs/") and path.endswith("/cancel"):
            return httpx.Response(200, json={"status": "cancelled"})
        if path.startswith("/jobs/"):
            jid = path.split("/")[2]
            return httpx.Response(200, json=self.job_rows.get(jid, {"status": "completed", "result": {"stdout": "gpu fine, disks fine"}}))
        return httpx.Response(404, json={})

    detail = ""

    def posted(self, suffix: str):
        return [c for c in self.calls if c[0] == "POST" and c[1].endswith(suffix)]


class ToolCase(unittest.TestCase):
    CONFIRM = ()  # most tests exercise the actions themselves; ConfirmMoreTest runs with the shipped defaults

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="apollo-tools-"))
        (self.tmp / "registry.yaml").write_text(REGISTRY)
        self.notes = self.tmp / "notes"
        self.notes.mkdir()
        self.world = FakeWorld()
        self.tools = AgentTools("http://gw", "tok", self.tmp / "registry.yaml", "http://gw/notify", self.notes,
                                self.tmp / "state", http=httpx.Client(transport=httpx.MockTransport(self.world.handler)),
                                confirm=self.CONFIRM)
        p = patch.object(agent_tools, "is_halted", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def say(self, text, apply=True):
        plan = self.tools.plan(text)
        if plan and apply and plan.apply:
            plan.apply()
        return plan

    def audit(self):
        f = self.tmp / "state" / "tool_audit.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


class ConversationTest(ToolCase):
    def test_ordinary_talk_is_not_a_tool_request(self):
        for text in ["how was your day", "i need to research my taxes someday", "tell me a joke", "what do you remember about me",
                     "the detective in that movie was great", "can you note that i like tea"]:
            self.assertIsNone(self.tools.plan(text), text)
        self.assertEqual(self.world.calls, [])


class NotifyTest(ToolCase):
    def test_nothing_is_sent_until_the_turn_commits(self):
        plan = self.tools.plan("notify me that the build finished")
        self.assertIn("Sending", plan.speak)
        self.assertEqual(self.world.posted("/notify"), [])  # planning is side-effect free (turns can be discarded)
        plan.apply()
        (call,) = self.world.posted("/notify")
        self.assertEqual(call[2], {"message": "the build finished", "source": "apollo"})

    def test_empty_notify_asks_what_to_say(self):
        plan = self.tools.plan("notify me")
        self.assertIn("What should", plan.speak)
        self.assertIsNone(plan.apply)

    def test_a_refusing_notifier_is_reported_on_the_next_turn(self):
        self.world.notifier_status = 500
        self.say("ping me to stretch")
        self.assertIn("refused", self.tools.take_notice())
        self.assertIsNone(self.tools.take_notice())
        self.assertEqual(self.audit()[-1]["outcome"][:6], "failed")


class ConfirmMoreTest(ToolCase):
    """The phone and existing notes are not touched until the owner says yes."""
    CONFIRM = None  # the defaults

    def test_defaults_cover_notify_and_note_writes(self):
        self.assertEqual(self.tools.confirm, agent_tools.CONFIRM_DEFAULT)
        self.assertEqual(agent_tools.CONFIRM_DEFAULT, {"notify", "notes_write"})

    def test_notify_reads_the_message_back_and_waits(self):
        plan = self.tools.plan("notify me that the build finished")
        self.assertIn("Send this to your phone: the build finished?", plan.speak)
        self.assertIn("Say yes", plan.speak)
        plan.apply()  # the turn commits: only the question is registered
        self.assertEqual(self.world.posted("/notify"), [])
        yes = self.tools.plan("yes")
        self.assertEqual(yes.speak, "On it.")
        yes.apply()
        (call,) = self.world.posted("/notify")
        self.assertEqual(call[2]["message"], "the build finished")

    def test_no_or_moving_on_sends_nothing(self):
        self.say("notify me that the build finished")
        self.assertIn("dropped", self.tools.plan("no").speak)
        self.say("notify me that the build finished")
        self.tools.plan("what is the weather like")  # moved on: never trapped in the question
        self.assertIsNone(self.tools.pending)
        self.assertEqual(self.world.posted("/notify"), [])

    def test_a_stale_yes_does_nothing(self):
        self.say("notify me hello there")
        self.tools.pending.expires = time.monotonic() - 1
        self.assertIsNone(self.tools.plan("yes"))
        self.assertEqual(self.world.posted("/notify"), [])

    def test_appending_to_a_note_waits_for_yes(self):
        (self.notes / "trip-plan.md").write_text("# trip plan\n\nkeep this")
        plan = self.say("add to note trip plan saying pack a charger")
        self.assertIn("Add this to trip plan: pack a charger?", plan.speak)
        self.assertNotIn("charger", (self.notes / "trip-plan.md").read_text())
        self.say("yes")
        self.assertIn("pack a charger", (self.notes / "trip-plan.md").read_text())

    def test_creating_onto_an_existing_note_waits_but_a_new_note_does_not(self):
        (self.notes / "trip-plan.md").write_text("# trip plan\n\nkeep this")
        plan = self.say("create a note called trip plan saying and this")
        self.assertIn("Say yes", plan.speak)
        self.assertNotIn("and this", (self.notes / "trip-plan.md").read_text())
        self.say("no")
        self.say("create a note called fresh idea saying try it")
        self.assertIn("try it", (self.notes / "fresh-idea.md").read_text())

    def test_the_old_immediate_behaviour_is_one_config_line_away(self):
        self.tools.confirm = frozenset()
        plan = self.tools.plan("notify me hi")
        self.assertIn("Sending", plan.speak)
        self.assertNotIn("Say yes", plan.speak)

    def test_confirmation_is_configurable_per_action(self):
        self.tools.confirm = frozenset({"notes_write"})
        self.assertIn("Sending", self.tools.plan("notify me hi").speak)

    def test_the_other_gates_are_unchanged(self):
        (self.notes / "old.md").write_text("x")
        self.assertIn("Say yes", self.tools.plan("delete the note old").speak)


class NotesTest(ToolCase):
    def write(self, name, body="hello"):
        (self.notes / name).write_text(body)

    def test_one_note_is_not_pluralised(self):
        self.write("groceries.md")
        self.assertIn("one note", self.tools.plan("list my notes").speak)

    def test_list_and_read(self):
        self.write("groceries.md", "# Groceries\n\nmilk and eggs")
        self.write("ideas.md", "# Ideas\n\nbuild a robot")
        self.assertIn("2 notes", self.tools.plan("list my notes").speak)
        self.assertIn("milk and eggs", self.tools.plan("read my note groceries").speak)
        self.assertIn("build a robot", self.tools.plan("read the latest note").speak)

    def test_create_append_and_read_back(self):
        self.say("create a note called trip plan saying book flights")
        text = (self.notes / "trip-plan.md").read_text()
        self.assertIn("# trip plan", text)
        self.assertIn("book flights", text)
        self.say("add to note trip plan saying pack a charger")
        self.assertIn("pack a charger", (self.notes / "trip-plan.md").read_text())

    def test_creating_an_existing_note_appends_instead_of_overwriting(self):
        self.write("trip-plan.md", "# trip plan\n\nkeep this")
        self.say("create a note called trip plan saying and this")
        text = (self.notes / "trip-plan.md").read_text()
        self.assertIn("keep this", text)
        self.assertIn("and this", text)

    def test_delete_needs_a_yes_and_moves_to_trash(self):
        self.write("old.md", "bye")
        plan = self.tools.plan("delete the note old")
        self.assertIn("Say yes", plan.speak)
        plan.apply()  # the question is registered when its turn commits
        self.assertTrue((self.notes / "old.md").exists())
        self.say("yes")
        self.assertFalse((self.notes / "old.md").exists())
        (trashed,) = list((self.notes / ".trash").glob("old-*.md"))
        self.assertEqual(trashed.read_text(), "bye")

    def test_anything_but_yes_cancels_the_delete(self):
        self.write("old.md", "keep")
        self.say("delete the note old")
        self.say("actually what time is it")
        self.say("yes")  # a stray yes later must not delete anything
        self.assertTrue((self.notes / "old.md").exists())

    def test_no_cancels_the_delete(self):
        self.write("old.md", "keep")
        self.say("delete the note old")
        self.assertIn("dropped", self.say("no").speak)
        self.assertTrue((self.notes / "old.md").exists())

    def test_a_late_yes_does_nothing(self):
        self.write("old.md", "keep")
        self.say("delete the note old")
        self.tools.pending.expires = time.monotonic() - 1
        self.assertIsNone(self.say("yes"))
        self.assertTrue((self.notes / "old.md").exists())

    def test_daily_notes_are_read_only_to_apollo(self):
        self.write("2026-10-31.md", "vself wrote this")
        for text in ["delete the note 2026-10-31", "add to note 2026-10-31 saying hi"]:
            self.assertIn("Virtual Self" if "delete" in text else "daily", self.tools.plan(text).speak)
        self.assertEqual((self.notes / "2026-10-31.md").read_text(), "vself wrote this")
        self.assertIn("vself wrote this", self.tools.plan("read the note 2026-10-31").speak)

    def test_paths_cannot_escape_the_notes_folder(self):
        victim = self.tmp / "secret.md"
        victim.write_text("secret")
        self.say("create a note called ../secret saying pwned")
        self.say("delete the note ../secret")
        self.say("yes")
        self.assertEqual(victim.read_text(), "secret")
        for p in self.notes.rglob("*"):
            self.assertTrue(p.resolve().is_relative_to(self.notes.resolve()))

    def test_a_symlinked_note_is_never_followed(self):
        target = self.tmp / "outside.md"
        target.write_text("outside")
        (self.notes / "link.md").symlink_to(target)
        plan = self.tools.plan("add to note link saying x")  # links are invisible to Apollo
        self.assertIsNone(plan.apply)
        self.assertEqual(target.read_text(), "outside")

    def test_ambiguous_names_ask_which(self):
        self.write("trip-plan.md")
        self.write("trip-budget.md")
        self.assertIn("Which one", self.tools.plan("read the note trip").speak)


class DumeRealAnswersTest(ToolCase):
    def test_filler_words_do_not_stop_a_machine_question_reaching_dume(self):
        self.world.chat_reply = {"reply": "Root is at sixty seven percent."}
        for q in ["Anyways tell me what's the exact status of the hard disk right now across the machine",
                  "okay so, anyway, how much disk space is left", "well can you tell me what is using the gpu"]:
            self.world.calls.clear()
            plan = self.say(q)
            self.assertIsNotNone(plan, q)
            self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1, q)

    def test_a_down_ai_brain_is_spoken_in_full(self):
        text = "AI brain: NOT running (no llama-server process; :8081 closed)\nVRAM used 5120 / 16384 MiB"
        out = agent_tools.summarize_snapshot(text)
        self.assertIn("the AI brain is NOT running", out)
        self.assertNotIn("is NOT.", out)

    def test_specific_questions_go_to_dume_itself_not_to_the_snapshot(self):
        self.world.chat_reply = {"reply": "Qwen 3.6 35B on llama-server, 8 gigabytes of VRAM."}
        for q in ["ask dummy what are the models loaded in the gpu", "ask dume which containers are running",
                  "ask dume to list the specific model names", "ask dume what is using the most memory"]:
            self.world.calls.clear()
            plan = self.say(q)
            self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1, q)
            self.assertEqual(self.world.posted("/agents/dume/jobs"), [], q)
            self.assertIn("Qwen", plan.speak, q)

    def test_a_plain_status_question_is_answered_by_dume_not_a_summary(self):
        self.world.chat_reply = {"reply": "All good."}
        self.assertEqual(self.say("ask dume how is everything").speak, "DUM-E says: All good.")
        self.assertEqual(self.world.posted("/agents/dume/jobs"), [])

    def test_a_garbled_opening_still_reaches_the_named_agent_with_the_question(self):
        self.world.chat_reply = {"reply": "Qwen 3.6 on llama-server."}
        plan = self.say("Kenya's dummy, what are the models loaded in the GPU?")
        (call,) = self.world.posted("/agents/dume/chat")
        self.assertIn("models loaded in the gpu", call[2]["message"].lower())
        self.assertIn("Qwen", plan.speak)

    def test_naming_an_agent_while_talking_about_it_is_not_a_request(self):
        for q in ["the detective in that movie was great", "who is the detective in that show", "what did the detective do"]:
            self.assertIsNone(self.tools.plan(q), q)
        self.assertEqual(self.world.calls, [])


class DumeVoiceTest(ToolCase):
    def test_every_way_the_mic_hears_dume_reaches_it(self):
        for name in ["dume", "dummy", "dum-e", "dumb e", "diom", "dium", "doom e", "dexta"]:
            self.world.calls.clear()
            plan = self.say(f"can you ask {name} what is going on in the system")
            self.assertIsNotNone(plan, name)
            path = "/agents/dume/chat" if name != "dexta" else "/agents/dexter/chat"
            self.assertEqual(len(self.world.posted(path)), 1, name)

    def test_a_status_question_is_status_not_a_change(self):
        for q in ["ask dummy what is going on in the system", "ask dume is everything holding up", "ask dume how the gpu is"]:
            self.assertNotIn("approve it in your terminal", self.say(q).speak, q)

    def test_real_change_requests_are_still_handed_back_to_the_terminal(self):
        for q in ["ask dume to restart nginx", "ask dummy to clean up docker images", "ask dume to reboot the machine"]:
            self.world.calls.clear()
            self.assertIn("approve it in your terminal", self.say(q).speak, q)
            self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1, q)
            self.assertEqual(self.world.posted("/agents/dume/jobs"), [], q)

    def test_extra_spellings_come_from_config(self):
        t = AgentTools("http://gw", "tok", self.tmp / "registry.yaml", "http://gw/notify", self.notes, self.tmp / "state",
                       http=httpx.Client(transport=httpx.MockTransport(self.world.handler)), aliases={"dume": ["zoomie"]})
        self.assertIsNotNone(t.plan("ask zoomie how the server is"))

    def test_near_misses_route_but_other_words_do_not(self):
        self.assertIsNone(self.tools.plan("ask mom about dinner"))
        self.assertIsNone(self.tools.plan("tell them about the plan"))


class NameRecognitionTest(ToolCase):
    ACT = {  # mangled, autocorrected or split names that must reach the right agent straight away
        "detective": ["the detective", "detective", "detectiv", "detecktive"],
        "dexter": ["dexter", "dexta", "dextor", "dexster", "decks ter", "deck ster"],
        "flowsmith": ["flowsmith", "flow smith", "flowsmyth", "floor smith", "flo smith"],
        "dume": ["dume", "dummy", "dum-e", "dumb e", "diom", "dium", "doom e", "dom e", "dum e"],
        "oda": ["oda", "o d a", "odah"],
    }

    def reaches(self, agent: str) -> bool:
        return bool(self.world.posted(f"/agents/{agent}/chat"))

    def test_every_listed_spelling_reaches_its_agent(self):
        for agent, names in self.ACT.items():
            for name in names:
                self.world.calls.clear()
                self.tools.pending = None
                plan = self.say(f"can you ask {name} about the latest status")
                if agent == "flowsmith":  # experimental: confirmed first, then asked
                    self.assertIn("experimental", plan.speak, name)
                    self.say("yes")
                self.assertTrue(self.reaches(agent), f"{name!r} did not reach {agent}")

    def test_other_openings_work_too(self):
        for text in ["tell dummy how the server is", "have the detective look into the port strike",
                     "check with dexta about nvidia", "talk to diom about the system"]:
            self.world.calls.clear()
            self.assertIsNotNone(self.say(text), text)
            self.assertTrue(self.world.calls, text)

    def test_a_doubtful_sound_alike_is_confirmed_before_anything_is_asked(self):
        plan = self.say("ask dumbie how the server is")
        self.assertIn("Did you mean DUM-E", plan.speak)
        self.assertEqual(self.world.calls, [])
        reply = self.say("yes")
        self.assertIn("DUM-E says", reply.speak)
        self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1)

    def test_a_no_to_the_guess_asks_nothing(self):
        self.say("ask dumbie how the server is")
        self.say("no")
        self.assertEqual(self.world.calls, [])

    def test_a_person_with_a_similar_name_costs_one_question_not_an_action(self):
        plan = self.say("ask dom about the budget")
        self.assertIn("Did you mean", plan.speak)
        self.assertEqual(self.world.calls, [])

    def test_ordinary_people_and_things_are_left_alone(self):
        for text in ["ask mom about dinner", "ask tom about the budget", "tell them about the plan", "start the car",
                     "run the dishwasher", "cancel the dentist job", "ask him to call me"]:
            self.assertIsNone(self.tools.plan(text), text)
        self.assertEqual(self.world.calls, [])

    def test_a_job_is_never_started_on_a_doubtful_name(self):
        for text in ["start dumbie nightly", "run dom job", "start odd a status"]:
            plan = self.tools.plan(text)
            self.assertTrue(plan is None or plan.apply is None, text)
        self.assertEqual(self.world.posted("/jobs"), [])

    def test_jobs_and_cancels_accept_garbled_names(self):
        plan = self.tools.plan("run the detectiv nightly")
        self.assertIn("Start the Detective nightly", plan.speak)
        plan.apply()
        self.say("yes")
        self.world.job_rows["job-1"] = {"status": "running"}
        self.assertIn("Cancel the Detective", self.tools.plan("cancel the detecktive job").speak)

    def test_the_sound_key_groups_what_it_should(self):
        self.assertEqual({agent_tools.skeleton(w) for w in ["dume", "dummy", "dum-e", "diom", "dumb"]}, {"dm"})
        self.assertEqual(agent_tools.skeleton("dexter"), agent_tools.skeleton("decks ter"))
        self.assertNotEqual(agent_tools.skeleton("dexter"), agent_tools.skeleton("detective"))


class RosterTest(ToolCase):
    def test_apollo_knows_its_agents(self):
        for q in ["what all agents you have access to", "are you aware which AI agent handles all the things in the system",
                  "which agents do you have", "do you have any tools"]:
            said = self.tools.plan(q).speak
            self.assertIn("Detective", said, q)
            self.assertIn("DUM-E", said, q)
        self.assertEqual(self.world.calls, [])  # a roster answer needs no gateway call

    def test_other_questions_about_tools_are_not_hijacked(self):
        for q in ["what tools should i buy for woodworking", "which agent in the movie is the villain"]:
            self.assertIsNone(self.tools.plan(q), q)

    def test_the_model_is_told_what_it_can_do(self):
        block = self.tools.capabilities_block()
        for needle in ("ask dume", "notify me", "list my notes", "never claim you asked unless you used that line", "Detective"):
            self.assertIn(needle, block)


class AskTest(ToolCase):
    def test_by_name_goes_to_that_agents_chat_and_speaks_the_reply(self):
        plan = self.say("ask the detective about the election audit")
        (call,) = self.world.posted("/agents/detective/chat")
        self.assertEqual(call[2]["message"], "the election audit")
        self.assertEqual(plan.speak, "Detective says: All quiet.")
        self.assertEqual(call[3]["authorization"], "Bearer tok")

    def test_by_topic_finance_goes_to_dexter_and_news_to_the_detective(self):
        self.say("research nvidia earnings")
        self.assertEqual(len(self.world.posted("/agents/dexter/chat")), 1)
        self.say("investigate the port strike")
        self.assertEqual(len(self.world.posted("/agents/detective/chat")), 1)

    def test_design_goes_to_oda(self):
        self.say("design a landing page for my portfolio")
        self.assertEqual(len(self.world.posted("/agents/oda/chat")), 1)

    def test_a_refusal_is_spoken_with_who_to_ask_instead(self):
        self.world.chat_reply = {"reply": '{"refused": "That is research.", "hand_off_to": "detective"}'}
        self.assertIn("Try detective", self.say("ask dexter about the strike").speak)

    def test_long_markdown_replies_are_made_speakable(self):
        self.world.chat_reply = {"reply": "## Report\n\n**Key** point one. " + "More detail here. " * 60 + "[src](http://x.y)"}
        spoken = self.say("ask the detective about it").speak
        self.assertNotIn("#", spoken)
        self.assertNotIn("**", spoken)
        self.assertLessEqual(len(spoken), agent_tools.AGENT_SPEAK_CHARS + 20)  # read out whole, up to a sane length

    def test_a_dume_change_becomes_a_proposal_the_owner_approves_in_the_terminal(self):
        self.world.chat_reply = {"reply": "Proposed restarting nginx as action 7. Run dume approve 7."}
        plan = self.say("ask dume to restart the nginx container")
        self.assertTrue(self.world.posted("/agents/dume/chat")[0][2]["message"].startswith("restart the nginx container"))
        self.assertIn("dume approve 7", plan.speak)
        self.assertIn("Nothing changes until you approve it in your terminal", plan.speak)
        self.assertEqual(self.world.posted("/agents/dume/jobs"), [])  # Apollo never starts a change itself

    def test_dume_status_is_dumes_own_words(self):
        self.world.chat_reply = {"reply": "Everything is fine. Disk is at sixty seven percent."}
        plan = self.say("how is the server")
        self.assertEqual(plan.speak, "DUM-E says: Everything is fine. Disk is at sixty seven percent.")
        self.assertEqual(self.world.posted("/agents/dume/jobs"), [])

    def test_when_dume_does_not_answer_the_snapshot_is_the_fallback(self):
        self.world.chat_status = 500
        plan = self.say("how is the server")
        self.assertIn("did not answer", plan.speak)
        self.assertIn("gpu fine", plan.speak)
        self.assertEqual(self.world.posted("/agents/dume/jobs")[0][2], {"payload": {"action": "snapshot"}})

    def test_flowsmith_is_confirmed_first_and_its_answer_arrives_as_a_notice(self):
        plan = self.say("ask flowsmith for a workflow that watches my feeds")
        self.assertIn("experimental", plan.speak)
        self.assertEqual(self.world.calls, [])
        self.say("yes")
        self.assertEqual(len(self.world.posted("/agents/flowsmith/chat")), 1)
        self.assertIn("replied", self.tools.take_notice())

    def test_gateway_down_is_spoken_not_raised(self):
        self.world.fail_connect = True
        self.assertIn("can't reach", self.say("ask the detective about x").speak)

    def test_open_circuit_is_explained(self):
        self.world.status, self.world.detail = 503, "detective is failing repeatedly; circuit open"
        self.assertIn("keeps failing", self.say("ask the detective about x").speak)

    def test_halted_gateway_is_explained(self):
        self.world.status, self.world.detail = 503, "halted by fortress-halt"
        self.assertIn("halted", self.say("ask the detective about x").speak)

    def test_an_agent_missing_from_the_registry_is_not_called(self):
        (self.tmp / "registry.yaml").write_text(REGISTRY.replace("id: oda", "id: other"))
        self.assertIn("don't have oda", self.say("ask oda to design a poster").speak)
        self.assertEqual(self.world.calls, [])


class JobTest(ToolCase):
    def test_starting_a_job_needs_a_yes_then_posts_with_an_idempotency_key_and_is_remembered(self):
        plan = self.tools.plan("run the detective nightly")
        self.assertIn("Say yes", plan.speak)
        self.assertEqual(self.world.calls, [])
        plan.apply()
        self.say("yes")
        (call,) = self.world.posted("/agents/detective/jobs")
        self.assertEqual(call[2], {"payload": {"type": "nightly"}})
        self.assertTrue(call[3]["idempotency-key"])
        saved = json.loads((self.tmp / "state" / "pending_jobs.json").read_text())
        self.assertEqual(saved[0]["job_id"], "job-1")

    def test_destructive_or_open_ended_job_types_cannot_be_started_by_voice(self):
        for text in ["run flowsmith delete workflow", "run flowsmith migrate pipeline", "run the detective inshorts sync",
                     "run oda tool"]:
            plan = self.tools.plan(text)
            self.assertTrue(plan is None or plan.apply is None, text)
        self.assertEqual(self.world.posted("/jobs"), [])
        self.assertFalse(any("delete" in c[1] for c in self.world.calls))

    def test_a_read_only_job_speaks_its_text_not_raw_json(self):
        self.world.job_rows["job-1"] = {"status": "completed", "result": {"stdout": "all systems ok", "exit_code": 0}}
        spoken = self.say("run oda status").speak
        self.assertIn("all systems ok", spoken)
        self.assertNotIn("{", spoken)

    def test_a_read_only_job_runs_without_asking(self):
        plan = self.say("run oda status")
        self.assertNotIn("Say yes", plan.speak)
        self.assertEqual(len(self.world.posted("/agents/oda/jobs")), 1)

    def test_status_reports_agents_and_remembered_jobs(self):
        self.say("run the detective nightly")
        self.say("yes")
        self.world.job_rows["job-1"] = {"status": "running"}
        said = self.say("what are the agents doing").speak
        self.assertIn("Detective nightly job is running", said)

    def test_cancel_asks_first_then_cancels(self):
        self.say("run the detective nightly")
        self.say("yes")
        self.world.job_rows["job-1"] = {"status": "running"}
        plan = self.tools.plan("cancel the detective job")
        self.assertIn("Cancel the Detective nightly job", plan.speak)
        plan.apply()
        self.assertEqual(self.world.posted("/cancel"), [])
        self.say("yes")
        self.assertEqual(len(self.world.posted("/cancel")), 1)

    def test_cancel_with_nothing_running_says_so(self):
        self.assertIn("no job", self.tools.plan("cancel the job").speak)


class HaltTest(ToolCase):
    def test_nothing_runs_while_halted(self):
        with patch.object(agent_tools, "is_halted", lambda: True):
            self.assertIsNone(self.tools.plan("ask the detective about x"))
            self.assertIsNone(self.tools.plan("notify me hello"))
        self.assertEqual(self.world.calls, [])


SNAPSHOT = """Uptime/load: 16:48:04 up  5:07,  2 users,  load average: 0.28, 0.53, 0.60
Memory:
total        used        free      shared  buff/cache   available
Mem:            30Gi       8.8Gi       302Mi       617Mi        22Gi        21Gi
Disks:
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  234G  202G   22G  91% /
/dev/sda1       1.9T  294G  1.6T  16% /mnt/expansion
GPU: NVIDIA GeForce RTX 5070 Ti | VRAM used 13958 / 16303 MiB, free 1874 MiB | util 0% | 37 C
AI brain: running (PID 508471, 2 min), model file x.gguf, ctx 262144, health ok
"""


class SnapshotTest(unittest.TestCase):
    def test_the_report_becomes_a_verdict_and_plain_sentences_without_decimals(self):
        said = agent_tools.summarize_snapshot(SNAPSHOT, cores=16)
        self.assertTrue(said.startswith("The machine is up, but the main disk is 91 percent full."))
        for part in ("The load is light", "about 21 gigabytes of memory are free", "using about 14 of its 16 gigabytes",
                     "the AI brain is running"):
            self.assertIn(part, said)
        self.assertNotRegex(said, r"\d\.\d")
        self.assertEqual(said.count("91 percent"), 1)

    def test_a_healthy_machine_gets_a_clean_verdict(self):
        ok = SNAPSHOT.replace(" 91% /", " 40% /")
        self.assertTrue(agent_tools.summarize_snapshot(ok, cores=16).startswith("The machine looks fine."))

    def test_trouble_is_named_in_the_verdict(self):
        bad = SNAPSHOT.replace("AI brain: running", "AI brain: stopped").replace("load average: 0.28", "load average: 30.0")
        said = agent_tools.summarize_snapshot(bad, cores=16)
        self.assertIn("the AI brain is stopped", said)
        self.assertIn("heavy load", said)

    def test_an_unreadable_report_is_not_invented(self):
        self.assertIsNone(agent_tools.summarize_snapshot("something odd"))


class NumbersTest(unittest.TestCase):
    def test_units_are_spoken_and_long_decimals_trimmed(self):
        out = agent_tools.smooth_numbers("Used 13.9GB of 16 GiB (87.54%), ~250 MB left, ratio 3.14159")
        self.assertIn("13.9 gigabytes", out)
        self.assertIn("16 gigabytes", out)
        self.assertIn("87.5 percent", out)
        self.assertIn("about 250 megabytes", out)
        self.assertIn("3.1", out)
        self.assertNotIn("3.14", out)


class AutoRouteTest(ToolCase):
    def test_machine_questions_reach_dume_without_saying_ask(self):
        for q in ["what is the status of the machine right now", "how is the server doing", "is the gpu busy",
                  "how much disk space is left", "what's eating my vram", "is everything holding up in the system",
                  "how's the fortress", "what is going on in the system"]:
            self.world.calls.clear()
            plan = self.tools.plan(q)
            self.assertIsNotNone(plan, q)
            self.assertEqual(len(self.world.posted("/agents/dume/jobs")) + len(self.world.posted("/agents/dume/chat")), 1, q)

    def test_finance_questions_reach_dexter_news_the_detective_design_oda(self):
        cases = {"/agents/dexter/chat": ["how is the nifty doing today", "what do you think about nvidia earnings"],
                 "/agents/detective/chat": ["what's the latest on the port strike", "any news about the election"],
                 "/agents/oda/chat": ["make me a landing page for my portfolio", "design a poster for the meetup"]}
        for path, qs in cases.items():
            for q in qs:
                self.world.calls.clear()
                self.assertIsNotNone(self.tools.plan(q), q)
                self.assertEqual(len(self.world.posted(path)), 1, q)

    def test_automation_asks_flowsmith_but_confirms_first(self):
        plan = self.tools.plan("automate my morning news digest")
        self.assertIn("experimental", plan.speak)
        self.assertEqual(self.world.calls, [])

    def test_everyday_talk_is_never_hijacked(self):
        for q in ["the market was so crowded today", "i bought groceries at the market", "how is my day going",
                  "what happened with me yesterday", "is it in stock", "what is the status of my relationship",
                  "how are you doing", "i love this system of government", "what's the latest on you"]:
            self.assertIsNone(self.tools.plan(q), q)
        self.assertEqual(self.world.calls, [])

    def test_a_slow_agent_returns_an_ack_first_when_asked_not_to_resolve(self):
        plan = self.tools.plan("what's the latest on the port strike", resolve=False)
        self.assertEqual(plan.ack, "Asking Detective.")
        self.assertEqual(self.world.calls, [])  # nothing slow has happened yet
        self.assertIn("Detective says", self.tools.finish(plan).speak)

    def test_a_failing_slow_agent_becomes_a_sentence_not_a_crash(self):
        self.world.fail_connect = True
        plan = self.tools.plan("what's the latest on the port strike", resolve=False)
        self.assertIn("can't reach", self.tools.finish(plan).speak)

    def test_ask_it_for_me_repeats_the_previous_request_through_the_router(self):
        self.tools.plan("what is the weather like")  # not a tool request, but it is what was said last
        self.assertIsNone(self.tools.plan("can you ask it for me"))
        self.tools.plan("how much disk space is left")
        self.world.calls.clear()
        again = self.tools.plan("can you ask it for me")
        self.assertIsNotNone(again)
        self.assertEqual(len(self.world.posted("/agents/dume/chat")), 1)


class SpeakableTest(unittest.TestCase):
    def test_trims_on_a_sentence_and_drops_markup(self):
        out = speakable("# Title\n\n`code` and **bold**. " + "Sentence number two goes here. " * 30)
        self.assertNotIn("`", out)
        self.assertTrue(out.endswith("."))
        self.assertLessEqual(len(out), 400)

    def test_links_are_not_read_out(self):
        self.assertEqual(speakable("see https://example.com/a/b now"), "see a link now")


if __name__ == "__main__":
    unittest.main()


class AsyncAnswersTest(ToolCase):
    """A slow agent must not hold the conversation: the turn ends, the answer is announced when it lands."""

    def test_a_quick_answer_is_the_replys_own_words(self):
        self.world.chat_reply = {"reply": "Qwen is loaded."}
        plan = self.say("ask dume which models are loaded")
        self.assertEqual(plan.speak, "DUM-E says: Qwen is loaded.")
        self.assertIsNone(self.tools.take_announcement())

    def test_a_slow_answer_ends_the_turn_then_is_announced_when_it_lands(self):
        self.tools.inline_wait_s = 0.05
        self.world.chat_delay = 0.4
        self.world.chat_reply = {"reply": "Found three things."}
        plan = self.say("ask the detective about the port strike")
        self.assertIn("still working on it", plan.speak)
        self.assertFalse(self.tools.has_announcement())
        deadline = time.monotonic() + 3
        while not self.tools.has_announcement() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.tools.take_announcement(), "Detective says: Found three things.")
        self.assertIsNone(self.tools.take_announcement())  # said once

    def test_a_late_failure_is_announced_too(self):
        self.tools.inline_wait_s = 0.05
        self.world.chat_delay = 0.3
        self.world.chat_status = 500
        self.say("ask the detective about the port strike")
        deadline = time.monotonic() + 3
        while not self.tools.has_announcement() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.tools.take_announcement())

    def test_a_long_job_is_announced_when_it_finishes(self):
        self.tools.job_poll_s = 0.05
        self.world.job_rows["job-1"] = {"status": "running"}
        self.say("run the detective nightly")
        self.say("yes")
        time.sleep(0.2)
        self.assertFalse(self.tools.has_announcement())
        self.world.job_rows["job-1"] = {"status": "completed", "result": {"stdout": "Wrote the briefing."}}
        deadline = time.monotonic() + 3
        while not self.tools.has_announcement() and time.monotonic() < deadline:
            time.sleep(0.02)
        said = self.tools.take_announcement()
        self.assertIn("Detective nightly job finished", said)
        self.assertIn("Wrote the briefing.", said)
        time.sleep(0.2)
        self.assertFalse(self.tools.has_announcement())  # once only

    def test_a_failed_job_is_announced(self):
        self.tools.job_poll_s = 0.05
        self.world.job_rows["job-1"] = {"status": "failed"}
        self.say("run the detective nightly")
        self.say("yes")
        deadline = time.monotonic() + 3
        while not self.tools.has_announcement() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIn("failed", self.tools.take_announcement())

    def test_a_job_that_finished_while_apollo_was_off_is_announced_on_the_next_start(self):
        self.tools.job_poll_s = 0.05
        self.tools._save_jobs([{"job_id": "job-9", "agent": "detective", "type": "nightly",
                                "started": agent_tools.datetime.now().astimezone().isoformat(timespec="seconds")},
                               {"job_id": "old", "agent": "detective", "type": "nightly", "announced": True,
                                "started": agent_tools.datetime.now().astimezone().isoformat(timespec="seconds")}])
        self.world.job_rows["job-9"] = {"status": "completed", "result": {"stdout": "Done overnight."}}
        self.tools.resume_jobs()
        deadline = time.monotonic() + 3
        while not self.tools.has_announcement() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIn("Done overnight.", self.tools.take_announcement())
        self.assertFalse(any(c[1] == "/jobs/old" for c in self.world.calls))


class PlaneToolTest(ToolCase):
    def setUp(self):
        super().setUp()
        from plane_client import PlaneClient

        self.plane = PlaneClient("http://plane/api/v1", "k", "my-workspace", "proj", project_label="MyBoard")
        self.plane.get_work_item = lambda ref: {"id": "wi-1", "sequence_id": 5, "name": "Fix login", "priority": "high",
                                                "description_html": "<p>Do it</p>"}
        self.plane.speak_list = lambda: "MyBoard has 1 ticket. Latest: LP-5 Fix login."
        self.plane.create_work_item = lambda name, desc="": {"id": "wi-2", "sequence_id": 6, "name": name}
        self.plane.ticket_label = PlaneClient.ticket_label
        self.plane.project_label = "MyBoard"
        self.plane.update_work_item = lambda ref, **kw: {"id": "wi-1", "sequence_id": 5, "name": kw.get("name", "Fix login")}
        self.plane.state_id = lambda s: "st-1"
        self.plane.delete_work_item = lambda ref: None
        self.tools.plane = self.plane

    def test_list_plane_tickets(self):
        plan = self.tools.plan("list plane tickets")
        self.assertIn("LP-5", plan.speak)

    def test_read_lp_number(self):
        plan = self.tools.plan("read ticket LP-5")
        self.assertIn("LP-5", plan.speak)

    def test_create_waits_for_yes(self):
        plan = self.tools.plan("create a plane ticket called test hook saying wire voice")
        self.assertIn("Say yes", plan.speak)
        self.tools.plan("create a plane ticket called test hook saying wire voice").apply()
        self.tools.plan("yes").apply()
        self.assertIn("Created LP-6", self.tools.take_announcement())

    def test_plane_marker_list(self):
        plan = self.tools.plane_command("list", "")
        self.assertIn("LP-5", plan.speak)

    def test_capabilities_mention_plane(self):
        block = self.tools.capabilities_block()
        self.assertIn("[[plane:list|]]", block)
        self.assertIn("Never say you cannot reach Plane", block)


class ReminderToolTest(ToolCase):
    def setUp(self):
        super().setUp()
        from reminder_client import ReminderClient, ReminderView

        self.rem = ReminderClient("http://127.0.0.1:8001", http=self.tools.http)
        self.rem.list = lambda enabled_only=False: [
            ReminderView(id=3, text="stand up", recurrence="daily", next_ts=1_700_000_000, enabled=True, hhmm="09:00")
        ]
        self.rem.speak_list = lambda: "You have 1 reminders: number 3, daily at 09:00, stand up."
        self.rem.add_in = lambda duration, text: ReminderView(
            id=4, text=text, recurrence="once", next_ts=1_700_000_100, enabled=True
        )
        self.rem.delete = lambda rid: None
        self.tools.reminders = self.rem

    def test_list_reminders_voice(self):
        plan = self.tools.plan("list my reminders")
        self.assertIn("1 reminders", plan.speak)

    def test_remind_in_confirms_then_creates(self):
        plan = self.tools.plan("remind me in 10m to drink water")
        self.assertIn("Remind you", plan.speak)
        self.tools.plan("remind me in 10m to drink water").apply()
        self.tools.plan("yes").apply()
        self.assertIn("reminder 4", self.tools.take_announcement().lower())
        self.assertTrue(any(a.get("tool") == "remind" for a in self.audit()))

    def test_remind_marker(self):
        plan = self.tools.reminder_command("list", "")
        self.assertIn("reminders", plan.speak)

    def test_plane_question_is_never_a_reminder(self):
        from plane_client import PlaneClient
        self.tools.plane = PlaneClient("http://plane/api/v1", "k", "my-workspace", "proj", project_label="MyBoard")
        self.tools.plane.speak_list = lambda: "MyBoard has 1 ticket."
        said = "Yeah. Can you tell me what are the plane projects that are open rightnow?"
        plan = self.tools.plan(said)
        self.assertEqual(plan.tool, "plane")
        self.assertFalse(self.tools.mentions_reminders(said))
        self.assertTrue(self.tools.mentions_reminders("remind me in ten minutes"))

    def test_voice_poll_is_throttled(self):
        calls = []
        self.rem.dequeue_voice = lambda limit=1: calls.append(1) or []
        for _ in range(50):
            self.tools.poll_voice_reminders()
        self.assertEqual(len(calls), 1)

    def test_capabilities_mention_reminders(self):
        block = self.tools.capabilities_block()
        self.assertIn("[[remind:list|]]", block)
        self.assertIn("Timed reminders", block)
