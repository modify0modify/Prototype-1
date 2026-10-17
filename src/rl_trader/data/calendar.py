"""Expected US equity trading-session dates for gap detection.

Uses weekdays minus a small built-in NYSE holiday set. This is intentionally
conservative for quality reporting — not a full exchange-calendar dependency.
Missing dates are **reported only**; they are never interpolated.
"""

from __future__ import annotations

from datetime import date, timedelta


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm — used for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th weekday (Mon=0) in month; n=-1 means last."""
    if n > 0:
        d = date(year, month, 1)
        while d.weekday() != weekday:
            d += timedelta(days=1)
        d += timedelta(weeks=n - 1)
        return d
    # last
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    """Approximate NYSE full-day holidays for ``year``."""
    holidays: set[date] = set()

    def observed(d: date) -> date:
        if d.weekday() == 5:  # Saturday -> Friday
            return d - timedelta(days=1)
        if d.weekday() == 6:  # Sunday -> Monday
            return d + timedelta(days=1)
        return d

    holidays.add(observed(date(year, 1, 1)))  # New Year
    holidays.add(_nth_weekday(year, 1, 0, 3))  # MLK
    holidays.add(_nth_weekday(year, 2, 0, 3))  # Presidents
    holidays.add(_easter(year) - timedelta(days=2))  # Good Friday
    holidays.add(_nth_weekday(year, 5, 0, -1))  # Memorial
    holidays.add(observed(date(year, 6, 19)))  # Juneteenth
    holidays.add(observed(date(year, 7, 4)))  # Independence
    holidays.add(_nth_weekday(year, 9, 0, 1))  # Labor
    holidays.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    holidays.add(observed(date(year, 12, 25)))  # Christmas
    return holidays


def expected_trading_days(start: date, end: date) -> list[date]:
    """Weekdays in ``[start, end]`` excluding approximate NYSE holidays."""
    if end < start:
        return []
    years = range(start.year, end.year + 1)
    holidays: set[date] = set()
    for year in years:
        holidays |= nyse_holidays(year)
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            days.append(current)
        current += timedelta(days=1)
    return days
