"""Conversation memory for ``ask``: what was asked and answered before, per conversation.

A follow-up ("and its minimum?") only means something next to the turns before it, so
``ask`` takes a conversation id. The server issues the ids. A conversation keeps its last
``max_turns`` question/answer pairs, and is forgotten after ``idle_seconds`` without a
question. Only the questions and the answers' text are kept, not the tool results. Each
answer is still grounded in the tools it called in its own turn, so a figure carried over
from an earlier answer has to be fetched again before it can pass the check.

The model is shown fewer turns than are kept (``for_model``): the last few that it answered.
Declined turns are kept for numbering but never shown: they hold nothing about the records,
and an off-topic exchange in the context was seen to pull later answers off course.

The memory is in-process: one server, and lost on restart. A client that sends an id the
server no longer knows gets a new conversation, and the result says so. Several replicas
would keep this in shared storage instead.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_MAX_TURNS = 10
DEFAULT_IDLE_SECONDS = 24 * 3600

#: Earlier turns the model sees with a new question.
TURNS_FOR_MODEL = 3


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
    session: str  # the turn's record id, so the next turn's record can point at it
    declined: bool = False


@dataclass
class Conversation:
    id: str
    turns: deque[Turn]
    count: int = 0  # turns ever answered, including the ones no longer kept
    last_used: float = field(default=0.0)
    doc_id: str | None = None  # the one file the conversation is about, if any

    @property
    def previous(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def for_model(self, n: int = TURNS_FOR_MODEL) -> list[tuple[str, str]]:
        """The last ``n`` answered (not declined) turns, as (question, answer), oldest first."""
        answered = [(t.question, t.answer) for t in self.turns if not t.declined]
        return answered[-n:] if n > 0 else []


class Conversations:
    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_turns = max_turns
        self.idle_seconds = idle_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, Conversation] = {}

    def resume(self, conversation_id: str | None, doc_id: str | None = None) -> Conversation:
        """The conversation with this id, or a new one if the id is None or unknown, or if
        it was about a different file than ``doc_id``: a conversation is about one file, and
        earlier turns about another file would only mislead."""
        with self._lock:
            self._forget_idle()
            found = self._items.get(conversation_id or "")
            if found is None or found.doc_id != doc_id:
                found = Conversation(uuid.uuid4().hex, deque(maxlen=self.max_turns), doc_id=doc_id)
                self._items[found.id] = found
            found.last_used = self.clock()
            return found

    def add(self, conversation: Conversation, turn: Turn) -> int:
        """Record an answered turn; returns its number in the conversation (from 1)."""
        with self._lock:
            conversation.turns.append(turn)
            conversation.count += 1
            conversation.last_used = self.clock()
            return conversation.count

    def _forget_idle(self) -> None:
        cutoff = self.clock() - self.idle_seconds
        for cid in [c.id for c in self._items.values() if c.last_used < cutoff]:
            del self._items[cid]
