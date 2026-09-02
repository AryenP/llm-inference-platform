import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

# https directly rather than relying on follow_redirects: arxiv 301s http, and
# following it on every page doubles the request count against a rate-limited API
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

DELAY_S = 3.0
MAX_RETRIES = 5

# arxiv returns 500 permanently for start >= 10000 on any single query, whatever
# totalResults claims. A corpus larger than that must be assembled from several
# narrower queries — hence the date windows below.
PAGE_CAP = 10000


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime | None


def parse_feed(xml: str) -> list[Paper]:
    root = ET.fromstring(xml)
    out = []
    for e in root.findall("a:entry", NS):
        raw_id = e.findtext("a:id", "", NS)
        # the id is a URL; the version suffix would fragment the primary key
        arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0] if raw_id else ""
        if not arxiv_id:
            continue
        updated = e.findtext("a:updated", None, NS)
        out.append(
            Paper(
                arxiv_id=arxiv_id,
                title=" ".join(e.findtext("a:title", "", NS).split()),
                abstract=" ".join(e.findtext("a:summary", "", NS).split()),
                categories=[c.attrib["term"] for c in e.findall("a:category", NS)],
                published=datetime.fromisoformat(e.findtext("a:published", "", NS)),
                updated=datetime.fromisoformat(updated) if updated else None,
            )
        )
    return out


def fetch(client, params, sleep=time.sleep) -> httpx.Response:
    # 5xx only: a malformed query is not going to succeed on the sixteenth try
    for attempt in range(MAX_RETRIES):
        r = client.get(API, params=params)
        if r.status_code < 500:
            break
        if attempt < MAX_RETRIES - 1:
            sleep(DELAY_S * 2**attempt)
    r.raise_for_status()
    return r


def windows(since: datetime, until: datetime, days: int) -> Iterator[tuple[datetime, datetime]]:
    hi = until
    while hi > since:
        lo = max(hi - timedelta(days=days), since)
        yield lo, hi
        hi = lo


def _stamp(d: datetime) -> str:
    return d.strftime("%Y%m%d%H%M")


def pages(
    query: str,
    limit: int,
    page_size: int = 200,
    window_days: int = 120,
    since: datetime | None = None,
    until: datetime | None = None,
    sleep=time.sleep,
    client=None,
) -> Iterator[list[Paper]]:
    since = since or datetime(2015, 1, 1, tzinfo=UTC)
    until = until or datetime.now(UTC)
    owned = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    seen = 0

    try:
        for lo, hi in windows(since, until, window_days):
            if seen >= limit:
                return
            scoped = f"({query}) AND submittedDate:[{_stamp(lo)} TO {_stamp(hi)}]"
            start = 0
            while start < PAGE_CAP and seen < limit:
                r = fetch(
                    client,
                    {
                        "search_query": scoped,
                        "start": start,
                        "max_results": min(page_size, limit - seen),
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                    sleep,
                )
                batch = parse_feed(r.text)
                sleep(DELAY_S)
                if not batch:
                    break
                seen += len(batch)
                start += page_size
                yield batch
            if start >= PAGE_CAP:
                print(
                    f"window {lo:%Y-%m-%d}..{hi:%Y-%m-%d} hit the {PAGE_CAP} cap; "
                    "narrow --window-days to reach the rest",
                    file=sys.stderr,
                )
    finally:
        if owned:
            client.close()
