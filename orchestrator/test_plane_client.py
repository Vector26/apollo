"""Plane API client (mocked HTTP)."""

import json
import unittest

import httpx

from plane_client import PlaneClient, PlaneError


class PlaneClientTest(unittest.TestCase):
    def setUp(self):
        self.calls: list[tuple] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append((request.method, request.url.path))
            path = request.url.path
            if path.endswith("/states/"):
                return httpx.Response(200, json=[{"id": "st-done", "name": "Done"}])
            if path.endswith("/work-items/") and request.method == "GET":
                return httpx.Response(200, json={
                    "results": [{"id": "wi-1", "sequence_id": 5, "name": "Fix login", "priority": "high",
                                 "description_html": "<p>Details</p>"}],
                    "next_cursor": None,
                })
            if path.endswith("/work-items/wi-1/") and request.method == "GET":
                return httpx.Response(200, json={"id": "wi-1", "sequence_id": 5, "name": "Fix login"})
            if path.endswith("/work-items/") and request.method == "POST":
                body = json.loads(request.content)
                return httpx.Response(201, json={"id": "wi-2", "sequence_id": 6, "name": body["name"]})
            if path.endswith("/work-items/wi-1/") and request.method == "PATCH":
                return httpx.Response(200, json={"id": "wi-1", "sequence_id": 5, "name": "Fix login", "state": "st-done"})
            if path.endswith("/work-items/wi-1/") and request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        self.client = PlaneClient(
            "http://plane/api/v1", "key", "my-workspace", "proj-1",
            http=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def test_list_and_resolve_lp_number(self):
        text = self.client.speak_list()
        self.assertIn("LP-5", text)
        item = self.client.resolve_ref("LP-5")
        self.assertEqual(item["id"], "wi-1")

    def test_create_update_delete(self):
        created = self.client.create_work_item("New task", "note")
        self.assertEqual(created["sequence_id"], 6)
        updated = self.client.update_work_item("LP-5", state="st-done")
        self.assertEqual(updated["state"], "st-done")
        self.client.delete_work_item("LP-5")
        self.assertIn(("DELETE", "/api/v1/workspaces/my-workspace/projects/proj-1/work-items/wi-1/"), self.calls)

    def test_state_lookup_is_fuzzy(self):
        self.assertEqual(self.client.state_id("done"), "st-done")

    def test_missing_ticket_raises(self):
        with self.assertRaises(PlaneError):
            self.client.resolve_ref("LP-999")


if __name__ == "__main__":
    unittest.main()
