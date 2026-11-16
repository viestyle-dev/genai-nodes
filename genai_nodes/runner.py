"""Pace LLM requests: one in flight, newest pending wins, interval and daily cap.

Requests only start from an explicit trigger. While one is running, a new
trigger replaces any waiting request instead of queuing; requests are spaced
by ``min_interval``; a per-day counter (persisted next to the graph's data)
caps spending; a 429 pauses requests for the time the provider asked for and
re-sends the newest request once.

Outputs follow core's origin rule: what ``submit`` reports happens inside the
trigger's dispatch, which its originator commits; what the timer and the
request task report is committed once at their end (``on_commit``).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import pathlib
import time
from typing import Callable

from .providers import Provider, ProviderError, RateLimited, Request, Response

logger = logging.getLogger(__name__)

DEFAULT_BACKOFF = 60.0


class DailyCounter:
    """Requests sent today, optionally persisted to a JSON file."""

    def __init__(self, path: pathlib.Path | None, key: str, today: Callable[[], str]):
        self._path, self._key, self._today = path, key, today
        self._day, self._count = today(), 0
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8")).get(key, {})
                if saved.get("day") == self._day:
                    self._count = int(saved.get("count", 0))
            except (ValueError, OSError):
                logger.warning("genai-nodes: cannot read %s", path)

    @property
    def count(self) -> int:
        self._roll()
        return self._count

    def _roll(self):
        day = self._today()
        if day != self._day:
            self._day, self._count = day, 0

    def increment(self):
        self._roll()
        self._count += 1
        if self._path is None:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
            data[self._key] = {"day": self._day, "count": self._count}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (ValueError, OSError):
            logger.warning("genai-nodes: cannot write %s", self._path)


class RequestRunner:
    def __init__(
        self,
        provider_for: Callable[[], Provider],
        on_result: Callable[[Request, Response], None],
        on_error: Callable[[str], None],
        on_busy: Callable[[bool], None],
        on_commit: Callable[[], None],
        settings: Callable[[], dict],
        counter: DailyCounter,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._provider_for = provider_for
        self._on_result, self._on_error, self._on_busy = on_result, on_error, on_busy
        self._on_commit = on_commit
        self._settings = settings
        self.counter = counter
        self._clock = clock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._pending: Request | None = None
        self._last_sent = -float("inf")
        self._backoff_until = -float("inf")

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop while running (between start and stop)."""
        return self._loop

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def stop(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._pending = None
        self._loop = None

    def submit(self, req: Request):
        if self._loop is None:
            self._on_error("not running (node is not active)")
            return
        self._pending = req  # newest wins
        self._kick()

    def _kick(self):
        if self._loop is None or self.busy or self._pending is None:
            return
        s = self._settings()
        now = self._clock()
        wait = max(
            float(s["min_interval_s"]) - (now - self._last_sent),
            self._backoff_until - now,
            0.0,
        )
        if wait > 0:
            if self._timer is None:
                self._timer = self._loop.call_later(wait, self._on_timer)
            return
        if self.counter.count >= int(s["max_requests_per_day"]):
            self._pending = None
            self._on_error(f"daily limit reached ({s['max_requests_per_day']} requests)")
            return
        req, self._pending = self._pending, None
        self.counter.increment()
        self._last_sent = now
        self._on_busy(True)
        self._task = self._loop.create_task(self._run(req))

    def _on_timer(self):
        self._timer = None
        self._kick()
        self._on_commit()

    def _error(self, req: Request, message: str):
        # Providers may echo request details in errors; never pass the key on
        if req.api_key:
            message = message.replace(req.api_key, "***")
        self._on_error(message)

    async def _run(self, req: Request):
        try:
            response = await asyncio.wait_for(self._provider_for().generate(req), timeout=req.timeout + 5.0)
            self._on_result(req, response)
        except asyncio.CancelledError:  # stop(): no outputs; stop() already cleared _task
            raise
        except RateLimited as e:
            self._backoff_until = self._clock() + (e.retry_after or DEFAULT_BACKOFF)
            if req.retries_left > 0 and self._pending is None:
                req.retries_left -= 1
                self._pending = req
            self._error(req, f"{e} (paused {e.retry_after or DEFAULT_BACKOFF:.0f} s)")
        except asyncio.TimeoutError:
            self._error(req, f"timed out after {req.timeout:.0f} s")
        except ProviderError as e:
            self._error(req, str(e))
        except Exception as e:  # never let a provider bug break the node
            # No traceback in the log: exception texts may contain the key
            self._error(req, f"{type(e).__name__}: {e}")
        self._on_busy(False)
        self._task = None
        self._kick()
        self._on_commit()


def today_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
