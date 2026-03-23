# ======================================================
# TIME WINDOW (SHARED CLOCK)
# ======================================================

from datetime import datetime, timedelta, UTC


def get_time_window(days_back: int):
    """
    Returns identical hourly-aligned time window
    for ALL data sources.
    """

    now = datetime.now(UTC)

    # hourly alignment
    end = now.replace(minute=0, second=0, microsecond=0)

    start = end - timedelta(days=days_back)

    return start, end