"""Conversation transcript storage and markdown note dumps with mermaid diagrams."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

NOTES_SYSTEM_PROMPT = """You are a note-taking assistant. Given a conversation transcript, produce a well-structured markdown document for humans to read later.

Include these sections:
1. # Session Notes — title with date/time
2. ## Summary — 2-4 sentence overview
3. ## Key Points — bullet list of main ideas discussed
4. ## Action Items — checkbox list of tasks/decisions (use - [ ] format)
5. ## Topic Flow — a mermaid flowchart showing how topics connected:
```mermaid
flowchart TD
    A[Topic1] --> B[Topic2]
```
6. ## Concept Map — a mermaid graph (DAG) of key concepts and relationships:
```mermaid
graph LR
    A[Concept] --> B[Related]
```
7. ## Decision Tree — if any decisions were discussed, a mermaid tree:
```mermaid
graph TD
    Root[Decision] --> Opt1[Option A]
    Root --> Opt2[Option B]
```

Use valid mermaid syntax. Keep node labels short. Output ONLY the markdown document, no preamble."""

DUMP_TRIGGERS = {"take notes", "dump notes", "save notes", "write notes"}


class NotesManager:
    def __init__(self, output_dir: str, llm_client, auto_dump_minutes: int = 5):
        self.output_dir = Path(output_dir)
        self.llm = llm_client
        self.auto_dump_minutes = auto_dump_minutes
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.output_dir / datetime.now().strftime("%Y-%m-%d")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.session_dir / f"transcript-{self.session_id}.jsonl"
        self.entries: List[dict] = []
        self._last_dump = datetime.now()

    def add(self, role: str, text: str):
        entry = {"ts": datetime.now().isoformat(), "role": role, "text": text}
        self.entries.append(entry)
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("[%s] %s", role, text[:80])

    def should_dump(self) -> bool:
        if not self.entries:
            return False
        elapsed = (datetime.now() - self._last_dump).total_seconds()
        return elapsed >= self.auto_dump_minutes * 60

    def is_dump_command(self, text: str) -> bool:
        lower = text.lower().strip()
        return any(t in lower for t in DUMP_TRIGGERS)

    def dump(self) -> Path:
        """Generate markdown notes file from transcript."""
        if not self.entries:
            logger.warning("No transcript entries to dump")
            return self.session_dir / f"session-{self.session_id}.md"

        transcript_text = "\n".join(
            f"[{e['ts']}] {e['role'].upper()}: {e['text']}" for e in self.entries
        )

        logger.info("Generating notes dump (%d entries)…", len(self.entries))
        md = self.llm.complete(NOTES_SYSTEM_PROMPT, transcript_text)

        if not md.strip().startswith("#"):
            md = f"# Session Notes — {self.session_id}\n\n{md}"

        out_path = self.session_dir / f"session-{self.session_id}.md"
        out_path.write_text(md)
        self._last_dump = datetime.now()
        logger.info("Notes written to %s", out_path)
        return out_path

    def validate_mermaid(self, md: str) -> List[str]:
        """Return list of mermaid blocks found (for observability)."""
        return re.findall(r"```mermaid\n(.*?)```", md, re.DOTALL)
