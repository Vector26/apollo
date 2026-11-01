"""Low-latency path tests: hedged LLM race, STT priority, DNS cache, speculative endpointing.

No GPU, audio device, or network required.
"""

import queue
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import audio_io
import dns_cache
from cli import Turn
from llm_client import LLMClient
from stt_stream import STTWorker


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _FakeStream:
    def __init__(self, tokens, first_delay, gap=0.0, fail=None):
        self.tokens, self.first_delay, self.gap, self.fail = tokens, first_delay, gap, fail
        self.closed = False

    def __iter__(self):
        time.sleep(self.first_delay)
        if self.fail:
            raise self.fail
        for t in self.tokens:
            if self.closed:
                return
            yield _chunk(t)
            time.sleep(self.gap)

    def close(self):
        self.closed = True


class _FakeCompletions:
    def __init__(self, plan):
        self.plan = plan
        self.streams = {}

    def create(self, model, **kw):
        s = self.plan[model]()
        self.streams[model] = s
        return s


def _client(plan, hedge_models):
    llm = LLMClient("http://x/v1", "primary", "sys", hedge=[{"model": m} for m in hedge_models])
    fake = _FakeCompletions(plan)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return llm, fake


class HedgedRaceTest(unittest.TestCase):
    def test_faster_route_wins_and_loser_is_closed(self):
        llm, fake = _client({
            "primary": lambda: _FakeStream(["slow ", "reply"], 0.4),
            "backup": lambda: _FakeStream(["fast ", "reply"], 0.05, gap=0.01),
        }, ["backup"])
        t0 = time.monotonic()
        out = "".join(llm.chat_stream("hi"))
        self.assertEqual(out, "fast reply")
        self.assertEqual(llm.last_route, "backup")
        self.assertLess(time.monotonic() - t0, 0.3)
        time.sleep(0.5)
        self.assertTrue(fake.streams["primary"].closed)
        self.assertEqual([m["role"] for m in llm.history], ["user", "assistant"])

    def test_failed_route_falls_back_to_other(self):
        llm, _ = _client({
            "primary": lambda: _FakeStream([], 0.01, fail=RuntimeError("dns")),
            "backup": lambda: _FakeStream(["ok"], 0.1),
        }, ["backup"])
        self.assertEqual("".join(llm.chat_stream("hi")), "ok")

    def test_all_routes_failing_raises(self):
        llm, _ = _client({
            "primary": lambda: _FakeStream([], 0.01, fail=RuntimeError("a")),
            "backup": lambda: _FakeStream([], 0.02, fail=RuntimeError("b")),
        }, ["backup"])
        with self.assertRaises(RuntimeError):
            "".join(llm.chat_stream("hi"))
        self.assertEqual(llm.history, [])

    def test_stt_drops_whisper_noise_segments(self):
        from types import SimpleNamespace
        import numpy as np
        from stt_stream import StreamingSTT
        seg = lambda text, nsp, lp: SimpleNamespace(text=text, no_speech_prob=nsp, avg_logprob=lp)
        stt = StreamingSTT.__new__(StreamingSTT)
        stt.language = "en"
        pcm = np.random.RandomState(0).randn(8000).astype(np.float32) * 0.02  # above the silence gate
        stt.model = SimpleNamespace(transcribe=lambda *a, **k: (
            [seg(" Thank you.", 0.9, -1.4), seg(" What time is it?", 0.1, -0.2)], None))
        self.assertEqual(stt.transcribe(pcm), "What time is it?")
        for noise, kept in ((" you", ""), (" Thanks for watching!", ""), (" You okay?", "You okay?")):
            stt.model = SimpleNamespace(transcribe=lambda *a, n=noise, **k: ([seg(n, 0.1, -0.3)], None))
            self.assertEqual(stt.transcribe(pcm), kept)

    def test_network_down_falls_back_to_local_brain(self):
        llm, _ = _client({
            "primary": lambda: _FakeStream([], 0.01, fail=OSError("Network is unreachable")),
            "backup": lambda: _FakeStream([], 0.01, fail=OSError("Network is unreachable")),
        }, ["backup"])
        local, _ = _client({"primary": lambda: _FakeStream(["local ", "reply"], 0.01)}, [])
        llm.fallback = local
        self.assertEqual("".join(llm.chat_stream("hi")), "local reply")
        self.assertEqual(llm.last_route, "fallback:primary")
        self.assertEqual(len(llm.history), 2)

    def test_talk_config_wires_local_fallback(self):
        from types import SimpleNamespace
        from cli import talk_llm_config
        cfg = {
            "llm": {"base_url": "http://127.0.0.1:8081/v1", "model": "local", "system_prompt": "s"},
            "brains": {"openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "remote",
                                      "fallback_to_local": True}},
        }
        prof = SimpleNamespace(profile_id="x", brain="openrouter", persona=None)
        llm_cfg, _ = talk_llm_config(cfg, prof)
        self.assertEqual(llm_cfg["fallback"]["model"], "local")
        self.assertNotIn("fallback", llm_cfg["fallback"])

    def test_uncommitted_and_abandoned_streams_leave_no_history(self):
        llm, _ = _client({"primary": lambda: _FakeStream(["a ", "b ", "c"], 0.0)}, [])
        "".join(llm.chat_stream("hi", commit=False))
        gen = llm.chat_stream("hi")
        next(gen)
        gen.close()
        self.assertEqual(llm.history, [])
        llm.commit("hi", "a b c")
        self.assertEqual(len(llm.history), 2)


class _FakeSTT:
    def __init__(self):
        self.calls = []

    def transcribe(self, pcm, beam_size=1):
        self.calls.append(len(pcm))
        time.sleep(0.05)
        return f"len{len(pcm)}"


class STTWorkerTest(unittest.TestCase):
    def test_final_wakes_immediately_and_carries_tag(self):
        evq: queue.Queue = queue.Queue()
        w = STTWorker(_FakeSTT(), evq)
        w.start()
        try:
            time.sleep(0.2)  # idle
            t0 = time.monotonic()
            w.submit_final(np.zeros(10, dtype=np.float32), 1.0, tag=7)
            ev = evq.get(timeout=1)
            self.assertLess(time.monotonic() - t0, 0.09)  # 50 ms fake transcribe + wake
            self.assertEqual(ev, ("final_text", "len10", 1.0, 7))
        finally:
            w.stop()

    def test_final_preempts_pending_partial(self):
        evq: queue.Queue = queue.Queue()
        stt = _FakeSTT()
        w = STTWorker(stt, evq)
        w.submit_partial(np.zeros(3, dtype=np.float32))
        w.submit_final(np.zeros(5, dtype=np.float32), 0.5, tag=1)
        w.start()
        try:
            ev = evq.get(timeout=1)
            self.assertEqual(ev[0], "final_text")
            time.sleep(0.2)
            self.assertEqual(stt.calls, [5])  # stale partial dropped
        finally:
            w.stop()


class DnsCacheTest(unittest.TestCase):
    def setUp(self):
        dns_cache._cache.clear()

    def test_serves_stale_when_resolver_fails(self):
        answers = [[("ok",)], socket.gaierror(socket.EAI_AGAIN, "temp")]

        def fake(host, port, *a, **k):
            r = answers.pop(0) if answers else socket.gaierror(socket.EAI_AGAIN, "temp")
            if isinstance(r, Exception):
                raise r
            return r

        with patch.object(dns_cache, "_real_getaddrinfo", fake), patch.object(dns_cache, "TTL_S", 0.0):
            self.assertEqual(dns_cache.cached_getaddrinfo("h", 443), [("ok",)])
            self.assertEqual(dns_cache.cached_getaddrinfo("h", 443), [("ok",)])  # stale, refresh fails
            time.sleep(0.1)
            self.assertEqual(dns_cache.cached_getaddrinfo("h", 443), [("ok",)])

    def test_first_lookup_retries_transient_failure(self):
        answers = [socket.gaierror(socket.EAI_AGAIN, "temp"), [("ok",)]]

        def fake(host, port, *a, **k):
            r = answers.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with patch.object(dns_cache, "_real_getaddrinfo", fake):
            self.assertEqual(dns_cache.cached_getaddrinfo("h2", 443), [("ok",)])


class _FakeInputStream:
    """Feeds a scripted amplitude sequence into the capture callback."""

    script: list = []

    def __init__(self, samplerate, channels, dtype, blocksize, latency, device, callback):
        self.cb, self.block = callback, blocksize

    def __enter__(self):
        def feed():
            for amp in self.script:
                self.cb(np.full((self.block, 1), amp, dtype=np.float32), self.block, None, None)
                time.sleep(0.001)
        threading.Thread(target=feed, daemon=True).start()
        return self

    def __exit__(self, *a):
        return False


class _FakeVad:
    def __call__(self, win, sr):
        return SimpleNamespace(item=lambda: 1.0 if float(win.abs().max()) > 0.05 else 0.0)

    def reset_states(self):
        pass


def _capture(script, speculate_ms):
    _FakeInputStream.script = script
    evq: queue.Queue = queue.Queue()
    stop = threading.Event()
    with patch.object(audio_io.torch.hub, "load", return_value=(_FakeVad(), (None,))):
        cap = audio_io.ContinuousCapture(
            sample_rate=16000, capture_rate=16000, silence_ms=200, min_speech_ms=100,
            partial_ms=10_000, event_queue=evq, stop_event=stop, speculate_ms=speculate_ms,
        )
    with patch.object(audio_io.sd, "InputStream", _FakeInputStream):
        cap.start()
        time.sleep(1.0)
        stop.set()
        cap._thread.join(timeout=2)
    kinds = []
    while not evq.empty():
        ev = evq.get()
        if ev[0] != "level":
            kinds.append(ev[0])
    return kinds


class SpeculativeEndpointTest(unittest.TestCase):
    # 2048-sample blocks at 16 kHz = 128 ms each
    def test_pause_precedes_speech_end(self):
        kinds = _capture([0.5] * 3 + [0.0] * 4, speculate_ms=96)
        self.assertEqual(kinds, ["speech_start", "pause", "speech_end"])

    def test_resume_cancels_pause(self):
        kinds = _capture([0.5] * 3 + [0.0] + [0.5] * 2 + [0.0] * 4, speculate_ms=96)
        self.assertEqual(kinds, ["speech_start", "pause", "resume", "pause", "speech_end"])

    def test_disabled_by_default(self):
        kinds = _capture([0.5] * 3 + [0.0] * 4, speculate_ms=0)
        self.assertEqual(kinds, ["speech_start", "speech_end"])


class TurnTest(unittest.TestCase):
    def test_wait_confirmed(self):
        stop = threading.Event()
        t = Turn(True, time.time())
        threading.Timer(0.05, t.confirmed.set).start()
        self.assertTrue(t.wait_confirmed(stop))
        t2 = Turn(True, time.time())
        threading.Timer(0.05, t2.cancel.set).start()
        self.assertFalse(t2.wait_confirmed(stop))
        self.assertNotEqual(t.id, t2.id)


if __name__ == "__main__":
    unittest.main()
