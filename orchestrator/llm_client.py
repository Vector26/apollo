"""OpenAI-compatible LLM client (local llama-server or OpenRouter)."""

import logging
import os
import queue
import re
import threading
from pathlib import Path
from typing import Generator, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

LOCAL_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def read_env_var(name: str, env_file: Optional[str] = None) -> Optional[str]:
    """Value of ``name`` from the process environment, else from a KEY=VALUE file."""
    if os.environ.get(name):
        return os.environ[name]
    if not env_file:
        return None
    path = Path(env_file).expanduser()
    if not path.is_file():
        return None
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.*)$")
    for line in path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip().strip("'\"") or None
    return None


class LLMClient:
    def __init__(self, base_url: str, model: str, system_prompt: str,
                 temperature: float = 0.7, max_tokens: int = 512,
                 api_key: Optional[str] = None, extra_body: Optional[dict] = None,
                 timeout_s: Optional[float] = None, max_retries: Optional[int] = None,
                 hedge: Optional[List[dict]] = None):
        kwargs: dict = {}
        if timeout_s is not None:
            kwargs["timeout"] = float(timeout_s)
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", **kwargs)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = LOCAL_EXTRA_BODY if extra_body is None else extra_body
        # extra routes raced against the primary; first token wins (tail-latency hedge)
        self.hedge = [
            (h["model"], h.get("extra_body", self.extra_body)) for h in (hedge or [])
        ]
        self.last_route: Optional[str] = None
        self.fallback: Optional["LLMClient"] = None  # used if every route fails before a token
        self.history: List[dict] = []

    @classmethod
    def from_config(cls, llm_cfg: dict, system_prompt: Optional[str] = None) -> "LLMClient":
        api_key = None
        key_env = llm_cfg.get("api_key_env")
        if key_env:
            api_key = read_env_var(key_env, llm_cfg.get("api_key_file"))
            if not api_key:
                raise RuntimeError(
                    f"{key_env} not set and not found in {llm_cfg.get('api_key_file') or '(no file)'}"
                )
        client = cls(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["model"],
            system_prompt=system_prompt or llm_cfg["system_prompt"],
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 512),
            api_key=api_key,
            extra_body=llm_cfg.get("extra_body"),
            timeout_s=llm_cfg.get("timeout_s"),
            max_retries=llm_cfg.get("max_retries"),
            hedge=llm_cfg.get("hedge"),
        )
        if llm_cfg.get("fallback"):
            client.fallback = cls.from_config(llm_cfg["fallback"], system_prompt=system_prompt)
        return client

    def reset(self):
        self.history.clear()

    def chat(self, user_message: str) -> str:
        """Send a message and return the full reply."""
        return "".join(self.chat_stream(user_message))

    def chat_stream(self, user_message: str, memory_block: str = "",
                    commit: bool = True) -> Generator[str, None, None]:
        """Stream reply tokens. History is only updated once the reply completes
        (and only if ``commit``); an abandoned stream leaves no trace."""
        system = self.system_prompt.rstrip()
        if memory_block:
            system = system + "\n\n" + memory_block
        messages = (
            [{"role": "system", "content": system}]
            + self.history
            + [{"role": "user", "content": user_message}]
        )
        full_reply = []
        try:
            for delta in self._race(messages):
                full_reply.append(delta)
                yield delta
        except Exception as e:
            if full_reply or self.fallback is None:
                raise
            logger.warning("All LLM routes failed (%s); falling back to %s", e, self.fallback.model)
            for delta in self.fallback._race(messages):
                full_reply.append(delta)
                yield delta
            self.last_route = f"fallback:{self.fallback.model}"
        reply_text = "".join(full_reply)
        if commit:
            self.commit(user_message, reply_text)
        logger.debug("LLM reply via %s (%d chars): %s", self.last_route, len(reply_text), reply_text[:80])

    def commit(self, user_message: str, reply_text: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply_text})

    def _open(self, model: str, extra_body: dict, messages: List[dict]):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            extra_body=extra_body,
        )

    @staticmethod
    def _deltas(stream) -> Generator[str, None, None]:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _race(self, messages: List[dict], on_open=None) -> Generator[str, None, None]:
        """Stream a reply. ``on_open(stream)`` hands every opened stream to the caller,
        who can close it from another thread to abandon a call blocked on the network."""
        routes = [(self.model, self.extra_body)] + self.hedge
        if len(routes) == 1:
            self.last_route = self.model
            stream = self._open(self.model, self.extra_body, messages)
            if on_open is not None:
                on_open(stream)
            try:
                yield from self._deltas(stream)
            finally:
                stream.close()
            return

        self.last_route = None
        events: queue.Queue = queue.Queue()
        lock = threading.Lock()
        winner: dict = {"idx": None}
        abandoned = threading.Event()
        opened: list = []

        def run(idx: int, model: str, extra_body: dict) -> None:
            stream = None
            try:
                stream = self._open(model, extra_body, messages)
                opened.append(stream)
                if on_open is not None:
                    on_open(stream)
                for delta in self._deltas(stream):
                    if abandoned.is_set():
                        return
                    with lock:
                        if winner["idx"] is None:
                            winner["idx"] = idx
                        elif winner["idx"] != idx:
                            return
                    events.put(("tok", idx, delta))
                events.put(("done", idx, None))
            except Exception as e:  # reported to the consumer; a loser's error is ignored
                events.put(("err", idx, e))
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

        for i, (model, extra_body) in enumerate(routes):
            threading.Thread(target=run, args=(i, model, extra_body), daemon=True,
                             name=f"llm-route-{i}").start()

        finished: dict = {}
        try:
            while True:
                kind, idx, val = events.get()
                if kind == "tok":
                    if self.last_route is None:
                        self.last_route = routes[idx][0]
                    yield val
                    continue
                finished[idx] = val
                if winner["idx"] == idx:
                    if kind == "err":
                        raise val
                    return
                if winner["idx"] is None and len(finished) == len(routes):
                    errors = [v for v in finished.values() if v is not None]
                    if len(errors) == len(routes):
                        raise errors[-1]
                    return
        finally:
            abandoned.set()
            for stream in list(opened):  # losers blocked mid-read are cut now, not at their next token
                try:
                    stream.close()
                except Exception:
                    pass

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        """One-shot completion (no history) — used for note generation."""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=max_tokens or 2048,
            extra_body=self.extra_body,
        )
        return r.choices[0].message.content or ""
