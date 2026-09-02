from datetime import UTC, datetime
from itertools import pairwise

import httpx
import pytest

from app.arxiv import MAX_RETRIES, PAGE_CAP, fetch, pages, windows

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/{id}v1</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>t</title><summary>s</summary><category term="cs.LG"/>
  </entry>
</feed>"""

EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None):
        self.calls.append(params or {})
        status, body = self.responses.pop(0)
        return httpx.Response(status, text=body, request=httpx.Request("GET", url))


def test_windows_tile_the_range_backwards_without_gaps():
    since = datetime(2024, 1, 1, tzinfo=UTC)
    until = datetime(2024, 12, 31, tzinfo=UTC)

    got = list(windows(since, until, 120))

    assert got[0][1] == until
    assert got[-1][0] == since
    assert all(a[0] == b[1] for a, b in pairwise(got))


def test_fetch_retries_5xx_then_succeeds():
    client = FakeClient([(503, ""), (503, ""), (200, EMPTY)])
    slept = []

    fetch(client, {}, sleep=slept.append)

    assert len(client.calls) == 3
    # exponential, off the arxiv delay
    assert slept == [3.0, 6.0]


def test_fetch_does_not_retry_4xx():
    client = FakeClient([(400, "")])

    with pytest.raises(httpx.HTTPStatusError):
        fetch(client, {}, sleep=lambda _: None)

    assert len(client.calls) == 1


def test_fetch_gives_up_after_max_retries():
    client = FakeClient([(500, "")] * MAX_RETRIES)

    with pytest.raises(httpx.HTTPStatusError):
        fetch(client, {}, sleep=lambda _: None)

    assert len(client.calls) == MAX_RETRIES


def test_pages_never_requests_beyond_the_pagination_cap():
    # every page full, so paging would run forever if the cap were not enforced
    client = FakeClient([(200, FEED.format(id=f"24{i:03d}.00001")) for i in range(400)])

    list(pages("q", limit=100_000, page_size=200, window_days=3650, client=client, sleep=lambda _: None))

    assert client.calls, "expected at least one request"
    assert max(c["start"] for c in client.calls) < PAGE_CAP


def test_pages_scopes_each_query_to_its_date_window():
    client = FakeClient([(200, EMPTY)] * 50)

    list(pages("cat:cs.LG", limit=10, window_days=120, client=client, sleep=lambda _: None))

    assert all("submittedDate:[" in c["search_query"] for c in client.calls)
    assert all(c["search_query"].startswith("(cat:cs.LG)") for c in client.calls)


def test_pages_stops_at_the_limit():
    client = FakeClient([(200, FEED.format(id="2401.00001"))] * 10)

    got = list(pages("q", limit=1, page_size=1, client=client, sleep=lambda _: None))

    assert sum(len(p) for p in got) == 1
    assert len(client.calls) == 1
