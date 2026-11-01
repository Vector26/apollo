import unittest

from announce import Announcer


class FakeTools:
    def __init__(self, items=()):
        self.items = list(items)

    def has_announcement(self):
        return bool(self.items)

    def take_announcement(self):
        return self.items.pop(0) if self.items else None


class FakeBrain:
    def __init__(self):
        self.noted = []

    def note_background(self, text):
        self.noted.append(text)


class AnnouncerTest(unittest.TestCase):
    def test_speaks_one_waiting_announcement_when_the_floor_is_free(self):
        said, brain = [], FakeBrain()
        a = Announcer(FakeTools(["A.", "B."]), brain, said.append, lambda: True)
        self.assertTrue(a.poll())
        self.assertEqual(said, ["A."])
        self.assertEqual(brain.noted, ["A."])
        self.assertTrue(a.poll())
        self.assertEqual(said, ["A.", "B."])
        self.assertFalse(a.poll())

    def test_never_interrupts_a_conversation(self):
        said = []
        tools = FakeTools(["Late answer."])
        busy = {"v": True}
        a = Announcer(tools, FakeBrain(), said.append, lambda: not busy["v"])
        self.assertFalse(a.poll())
        self.assertEqual(said, [])
        self.assertEqual(tools.items, ["Late answer."])  # kept, not lost
        busy["v"] = False
        self.assertTrue(a.poll())
        self.assertEqual(said, ["Late answer."])

    def test_no_tools_or_a_brain_without_notes_is_fine(self):
        self.assertFalse(Announcer(None, None, print, lambda: True).poll())
        said = []
        self.assertTrue(Announcer(FakeTools(["x"]), object(), said.append, lambda: True).poll())
        self.assertEqual(said, ["x"])


if __name__ == "__main__":
    unittest.main()
