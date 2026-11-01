# Reminder service contract

Apollo's reminders are stored and scheduled by a separate service. Apollo only talks to it over HTTP through `orchestrator/reminder_client.py`. Any service that implements the endpoints below will work. The reference server is not part of this repository.

Base URL is `tools.reminders.base_url` in `config.yaml`; `tools.reminders.tz` is the IANA timezone used to speak times back.

## Reminder object

```json
{ "id": 7, "text": "stand up", "recurrence": "once", "next_ts": 1793000000,
  "enabled": true, "hhmm": null, "weekdays": null }
```

`next_ts` is epoch seconds. `hhmm` and `weekdays` are set for recurring reminders.

## Endpoints

| Method and path | Body or query | Returns |
|---|---|---|
| `GET /reminders` | `?enabled_only=true` optional | `{"reminders": [reminder, ...]}` |
| `GET /reminders/{id}` | | one reminder, `404` if unknown |
| `POST /reminders/in` | `{"duration": "10m", "text": "..."}` | reminder |
| `POST /reminders/at` | `{"when": "<local time>", "text": "..."}` | reminder |
| `POST /reminders/daily` | `{"hhmm": "09:00", "text": "..."}` | reminder |
| `POST /reminders/weekly` | `{"days": "mon,wed", "hhmm": "09:00", "text": "..."}` | reminder |
| `PATCH /reminders/{id}` | any of `text`, `enabled`, `in_duration`, `at_local` | reminder |
| `DELETE /reminders/{id}` | | empty |
| `GET /reminders/voice/dequeue` | `?limit=3` | `{"messages": ["...", ...]}` |

Errors: the client maps `404` to "couldn't find that reminder" and any other status of 400 or above to a spoken refusal that includes the server's `detail` field.

## Voice delivery

When a reminder comes due, the service should (a) notify whatever channel you use (the reference server sends Telegram) and (b) push a spoken line onto a queue. Apollo polls `GET /reminders/voice/dequeue` (throttled to once every two seconds) and speaks whatever it returns when you are not mid-conversation. Dequeue must mark lines delivered, so each is spoken once. The reference server keeps this queue in SQLite, so a line queued while Apollo is off is still spoken after it starts.

Reminders live in the service's database, not in Apollo, so they survive Apollo restarts.
