"""HTTP client for Telemon timed reminders (same store as the Telegram bot)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

import httpx


class ReminderError(Exception):
    def __init__(self, spoken: str) -> None:
        self.spoken = spoken
        super().__init__(spoken)


@dataclass
class ReminderView:
    id: int
    text: str
    recurrence: str
    next_ts: int
    enabled: bool
    hhmm: Optional[str] = None
    weekdays: Optional[List[int]] = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "ReminderView":
        return cls(
            id=int(raw["id"]),
            text=str(raw["text"]),
            recurrence=str(raw["recurrence"]),
            next_ts=int(raw["next_ts"]),
            enabled=bool(raw.get("enabled", True)),
            hhmm=raw.get("hhmm"),
            weekdays=raw.get("weekdays"),
        )


class ReminderClient:
    def __init__(self, base_url: str, http: httpx.Client | None = None, tz: str = "UTC") -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=10.0)
        self.tz = ZoneInfo(tz)

    def _req(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            r = self.http.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.HTTPError:
            raise ReminderError("I couldn't reach the reminder service.")
        if r.status_code == 404:
            raise ReminderError("I couldn't find that reminder.")
        if r.status_code >= 400:
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
            raise ReminderError(f"The reminder service refused that: {detail}")
        return r.json() if r.content else {}

    def list(self, enabled_only: bool = False) -> List[ReminderView]:
        q = "?enabled_only=true" if enabled_only else ""
        data = self._req("GET", f"/reminders{q}")
        return [ReminderView.from_api(x) for x in data.get("reminders", [])]

    def get(self, rem_id: int) -> ReminderView:
        return ReminderView.from_api(self._req("GET", f"/reminders/{rem_id}"))

    def add_in(self, duration: str, text: str) -> ReminderView:
        return ReminderView.from_api(self._req("POST", "/reminders/in", json={"duration": duration, "text": text}))

    def add_at(self, when_local: str, text: str) -> ReminderView:
        return ReminderView.from_api(self._req("POST", "/reminders/at", json={"when": when_local, "text": text}))

    def add_daily(self, hhmm: str, text: str) -> ReminderView:
        return ReminderView.from_api(self._req("POST", "/reminders/daily", json={"hhmm": hhmm, "text": text}))

    def add_weekly(self, days: str, hhmm: str, text: str) -> ReminderView:
        return ReminderView.from_api(
            self._req("POST", "/reminders/weekly", json={"days": days, "hhmm": hhmm, "text": text})
        )

    def patch(
        self,
        rem_id: int,
        *,
        text: str | None = None,
        enabled: bool | None = None,
        in_duration: str | None = None,
        at_local: str | None = None,
    ) -> ReminderView:
        body = {k: v for k, v in {
            "text": text,
            "enabled": enabled,
            "in_duration": in_duration,
            "at_local": at_local,
        }.items() if v is not None}
        return ReminderView.from_api(self._req("PATCH", f"/reminders/{rem_id}", json=body))

    def delete(self, rem_id: int) -> None:
        self._req("DELETE", f"/reminders/{rem_id}")

    def dequeue_voice(self, limit: int = 3) -> List[str]:
        try:
            r = self.http.get(f"{self.base_url}/reminders/voice/dequeue", params={"limit": limit}, timeout=5.0)
        except httpx.HTTPError:
            return []
        if r.status_code >= 400:
            return []
        return list(r.json().get("messages") or [])

    def _fmt_when(self, epoch: int) -> str:
        dt = datetime.fromtimestamp(epoch, ZoneInfo("UTC")).astimezone(self.tz)
        return dt.strftime("%Y-%m-%d %H:%M")

    def speak_list(self) -> str:
        rems = [r for r in self.list() if r.enabled]
        if not rems:
            return "You have no active reminders."
        parts = []
        for r in rems[:8]:
            if r.recurrence == "once":
                rec = "once"
            elif r.recurrence == "daily":
                rec = f"daily at {r.hhmm}"
            else:
                rec = f"weekly at {r.hhmm}"
            parts.append(f"number {r.id}, {rec}, {self._fmt_when(r.next_ts)}, {r.text}")
        extra = f" and {len(rems) - 8} more" if len(rems) > 8 else ""
        return f"You have {len(rems)} reminders{extra}: " + "; ".join(parts) + "."

    def speak_one(self, r: ReminderView) -> str:
        status = "on" if r.enabled else "off"
        return f"Reminder {r.id}, {status}, {self._fmt_when(r.next_ts)}, {r.text}."
