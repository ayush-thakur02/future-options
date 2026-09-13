"""NSE trading session helpers.

Everything here works in ``Asia/Kolkata``. Holiday handling is deliberately
data-driven: rather than hard-coding a holiday list that silently rots, the
calendar keys off weekends, session hours, and an optional user-maintained
holiday file. In practice an absent bar simply never appears in the dataset,
so backtests stay correct either way.

The clock is deliberately injectable: the runtime passes a calendar rather than
reading the wall clock directly, so a replay can be driven at any speed and the
session logic still behaves.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)

SESSION_MINUTES = 375  # 09:15 -> 15:30


class TradingCalendar:
    """Session clock and holiday bookkeeping for NSE."""

    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays: set[date] = holidays or set()

    @classmethod
    def from_file(cls, path: Path) -> TradingCalendar:
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text())
        parsed = {date.fromisoformat(item) for item in payload.get("holidays", [])}
        return cls(parsed)

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def now(self) -> datetime:
        return datetime.now(IST)

    def is_open(self, moment: datetime | None = None) -> bool:
        moment = (moment or self.now()).astimezone(IST)
        if not self.is_trading_day(moment.date()):
            return False
        return SESSION_OPEN <= moment.time() <= SESSION_CLOSE

    def is_pre_open(self, moment: datetime | None = None) -> bool:
        moment = (moment or self.now()).astimezone(IST)
        if not self.is_trading_day(moment.date()):
            return False
        return PRE_OPEN_START <= moment.time() < SESSION_OPEN

    def session_progress(self, moment: datetime | None = None) -> float:
        """Fraction of the trading session elapsed, clamped to [0, 1]."""
        moment = (moment or self.now()).astimezone(IST)
        if not self.is_trading_day(moment.date()):
            return 0.0
        elapsed = (moment.hour * 60 + moment.minute) - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
        return min(max(elapsed / SESSION_MINUTES, 0.0), 1.0)

    def minutes_to_close(self, moment: datetime | None = None) -> int:
        moment = (moment or self.now()).astimezone(IST)
        close_dt = datetime.combine(moment.date(), SESSION_CLOSE, tzinfo=IST)
        return max(int((close_dt - moment).total_seconds() // 60), 0)

    def next_open(self, moment: datetime | None = None) -> datetime:
        moment = (moment or self.now()).astimezone(IST)
        candidate = moment.date()
        if moment.time() >= SESSION_OPEN:
            candidate = candidate + timedelta(days=1)
        for _ in range(15):
            if self.is_trading_day(candidate):
                return datetime.combine(candidate, SESSION_OPEN, tzinfo=IST)
            candidate = candidate + timedelta(days=1)
        raise RuntimeError("no trading day found within two weeks")

    def last_open(self, moment: datetime | None = None) -> datetime:
        moment = (moment or self.now()).astimezone(IST)
        candidate = moment.date()
        if not self.is_trading_day(candidate) or moment.time() < SESSION_OPEN:
            candidate = candidate - timedelta(days=1)
        for _ in range(15):
            if self.is_trading_day(candidate):
                return datetime.combine(candidate, SESSION_OPEN, tzinfo=IST)
            candidate = candidate - timedelta(days=1)
        raise RuntimeError("no trading day found within two weeks")

    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        return (
            datetime.combine(day, SESSION_OPEN, tzinfo=IST),
            datetime.combine(day, SESSION_CLOSE, tzinfo=IST),
        )

    def previous_trading_day(self, day: date) -> date:
        candidate = day - timedelta(days=1)
        for _ in range(15):
            if self.is_trading_day(candidate):
                return candidate
            candidate = candidate - timedelta(days=1)
        return candidate


def to_ist(moment: datetime) -> datetime:
    """Attach or convert to IST."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=IST)
    return moment.astimezone(IST)
