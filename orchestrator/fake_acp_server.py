#!/usr/bin/env python3
"""A stand-in for `cursor-agent acp` used only by the tests. What it does is chosen by a keyword in the owner's
line of the prompt (SLOW, TOOL, PERM, LONG, MARKDOWN, MARKER, NETFAIL, CRASH); everything it receives is logged as JSON
lines to $FAKE_LOG so tests can assert what was sent."""
import json
import os
import sys
import threading
import time

LOG = os.environ.get("FAKE_LOG")
cancelled: set = set()
sessions = 0
perm_answers: dict = {}
out_lock = threading.Lock()


def log(**kw):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(kw) + "\n")


def send(obj):
    with out_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def chunk(sid, text):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {
        "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}}})


def prompt(rid, p):
    sid = p["sessionId"]
    text = p["prompt"][0]["text"]
    log(event="prompt", sid=sid, text=text)
    owner = text.rsplit("Owner:", 1)[-1]
    if "CRASH" in owner:
        os._exit(1)
    if "SLOW" in owner:
        for _ in range(60):
            if sid in cancelled:
                break
            time.sleep(0.05)
    if "TOOL" in owner:
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {
            "sessionUpdate": "tool_call", "toolCallId": "t1", "title": "grep", "status": "pending"}}})
        for _ in range(40):
            if sid in cancelled:
                break
            time.sleep(0.05)
    if "PERM" in owner:
        send({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission", "params": {
            "sessionId": sid, "toolCall": {"title": "shell"},
            "options": [{"optionId": "ok", "kind": "allow_once"}, {"optionId": "no", "kind": "reject_once"}]}})
        for _ in range(40):
            if 900 in perm_answers:
                break
            time.sleep(0.05)
        log(event="permission_answer", answer=perm_answers.get(900))
    if sid in cancelled:
        cancelled.discard(sid)
        send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
        return
    if "LONG" in owner:
        pieces = [f"Sentence number {i} is right here. " for i in range(40)]
    elif "MARKDOWN" in owner:
        pieces = ["**Bold", "** and `code`", "\n- item one\n", "# Heading\n", "done."]
    elif "MARKER" in owner:
        pieces = ["[[ask:dume|how much", " ram is in use]]"]
    elif "NETFAIL" in owner:
        pieces = ["Error: RetriableError:\n", "[unavailable] getaddrinfo EAI_AGAIN agentn.global.api5.cursor.sh"]
    elif "HEDGE" in owner:
        pieces = ["Composer ", "here."]
    else:
        pieces = ["Hello ", "there. ", "Second sentence."]
    for piece in pieces:
        if sid in cancelled:
            cancelled.discard(sid)
            log(event="stopped_mid_stream")
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
            return
        chunk(sid, piece)
        time.sleep(0.02)
    send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})


def main():
    global sessions
    for line in sys.stdin:
        m = json.loads(line)
        meth, rid = m.get("method"), m.get("id")
        if meth is None and rid == 900:
            perm_answers[900] = m.get("result")
            continue
        if meth == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1, "authMethods": [{"id": "cursor_login"}]}})
        elif meth == "authenticate":
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif meth == "session/new":
            sessions += 1
            log(event="session_new", cwd=m["params"].get("cwd"))
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "sessionId": f"s{sessions}", "models": {"currentModelId": os.environ.get("FAKE_MODEL", "composer-2.5[fast=true]")}}})
        elif meth in ("session/set_mode", "session/set_model"):
            log(event=meth.split("/")[1], **{k: v for k, v in m["params"].items() if k != "sessionId"})
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif meth == "session/cancel":
            log(event="cancel", sid=m["params"]["sessionId"])
            cancelled.add(m["params"]["sessionId"])
        elif meth == "session/prompt":
            threading.Thread(target=prompt, args=(rid, m["params"]), daemon=True).start()


if __name__ == "__main__":
    main()
