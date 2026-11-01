"""Plane.so REST client for Apollo (self-hosted ticketing CRUD)."""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
REF_RE = re.compile(r"^(?:LP[- ]?)?(\d+)$", re.I)


class PlaneError(Exception):
    def __init__(self, spoken: str, status: int | None = None) -> None:
        super().__init__(spoken)
        self.spoken = spoken
        self.status = status


class PlaneClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        workspace_slug: str,
        project_id: str,
        project_label: str = "Plane",
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workspace_slug = workspace_slug
        self.project_id = project_id
        self.project_label = project_label
        self.http = http or httpx.Client(timeout=20.0)
        self._states: dict[str, str] | None = None  # lower name -> id
        self._index_at = 0.0
        self._by_seq: dict[int, dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.http.close()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def _project_path(self, project_id: str | None = None) -> str:
        pid = project_id or self.project_id
        return f"/workspaces/{self.workspace_slug}/projects/{pid}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            r = self.http.request(method, url, headers=self._headers(), **kwargs)
        except httpx.HTTPError as e:
            raise PlaneError("I couldn't reach Plane on the network.") from e
        if r.status_code == 401:
            raise PlaneError("Plane refused the API key.", 401)
        if r.status_code == 404:
            raise PlaneError("That ticket or project wasn't found in Plane.", 404)
        if r.status_code >= 400:
            raise PlaneError(f"Plane returned an error ({r.status_code}).", r.status_code)
        if r.status_code == 204:
            return None
        if not r.content:
            return None
        return r.json()

    def refresh_index(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._index_at < 45 and self._by_seq:
            return
        self._by_seq.clear()
        self._by_id.clear()
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"per_page": 100}
            if cursor:
                params["cursor"] = cursor
            body = self._request("GET", f"{self._project_path()}/work-items/", params=params)
            results = (body or {}).get("results") or []
            for item in results:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                self._by_id[str(item["id"])] = item
                seq = item.get("sequence_id")
                if seq is not None:
                    self._by_seq[int(seq)] = item
            cursor = (body or {}).get("next_cursor")
            if not cursor or not results:
                break
        self._index_at = time.monotonic()

    def resolve_ref(self, ref: str) -> dict[str, Any]:
        ref = ref.strip()
        if UUID_RE.match(ref):
            self.refresh_index()
            if ref in self._by_id:
                return self._by_id[ref]
            return self._request("GET", f"{self._project_path()}/work-items/{ref}/")
        m = REF_RE.match(ref.replace(" ", ""))
        if not m:
            raise PlaneError(f"I didn't understand the ticket reference {ref!r}. Use LP-5 or the UUID.")
        self.refresh_index(force=True)
        seq = int(m.group(1))
        if seq not in self._by_seq:
            raise PlaneError(f"I couldn't find ticket LP-{seq} on {self.project_label}.")
        return self._by_seq[seq]

    def list_work_items(self, limit: int = 12) -> list[dict[str, Any]]:
        self.refresh_index(force=True)
        items = sorted(self._by_seq.values(), key=lambda x: int(x.get("sequence_id") or 0), reverse=True)
        return items[:limit]

    def get_work_item(self, ref: str) -> dict[str, Any]:
        return self.resolve_ref(ref)

    def create_work_item(self, name: str, description: str = "", **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name.strip()[: 255]}
        if description.strip():
            payload["description_html"] = f"<p>{html.escape(description.strip())}</p>"
        payload.update(extra)
        out = self._request("POST", f"{self._project_path()}/work-items/", json=payload)
        self.refresh_index(force=True)
        return out

    def update_work_item(self, ref: str, **fields: Any) -> dict[str, Any]:
        item = self.resolve_ref(ref)
        wid = item["id"]
        out = self._request("PATCH", f"{self._project_path()}/work-items/{wid}/", json=fields)
        self.refresh_index(force=True)
        return out

    def delete_work_item(self, ref: str) -> None:
        item = self.resolve_ref(ref)
        self._request("DELETE", f"{self._project_path()}/work-items/{item['id']}/")
        self.refresh_index(force=True)

    def states(self) -> dict[str, str]:
        if self._states is not None:
            return self._states
        rows = self._request("GET", f"{self._project_path()}/states/") or []
        self._states = {str(r.get("name", "")).lower(): str(r["id"]) for r in rows if r.get("id")}
        return self._states

    def state_id(self, name: str) -> str:
        key = name.strip().lower()
        states = self.states()
        if key in states:
            return states[key]
        for label, sid in states.items():
            if key in label or label in key:
                return sid
        raise PlaneError(f"I couldn't find a Plane state called {name!r}.")

    @staticmethod
    def ticket_label(item: dict[str, Any]) -> str:
        seq = item.get("sequence_id")
        prefix = f"LP-{seq}" if seq is not None else "ticket"
        return f"{prefix} {item.get('name', '').strip()}"

    @staticmethod
    def describe_item(item: dict[str, Any], limit: int = 380) -> str:
        label = PlaneClient.ticket_label(item)
        bits = [label]
        pri = item.get("priority")
        if pri and pri != "none":
            bits.append(f"priority {pri}")
        desc = re.sub(r"<[^>]+>", " ", item.get("description_html") or "")
        desc = re.sub(r"\s+", " ", desc).strip()
        if desc:
            bits.append(desc[:200] + ("…" if len(desc) > 200 else ""))
        text = ". ".join(bits)
        return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"

    def speak_list(self, limit: int = 8) -> str:
        items = self.list_work_items(limit=limit)
        if not items:
            return f"{self.project_label} has no tickets yet."
        names = ", ".join(self.ticket_label(i) for i in items[:limit])
        total = len(self._by_seq)
        head = f"{self.project_label} has {total} ticket{'s' if total != 1 else ''}."
        return f"{head} Latest: {names}."
