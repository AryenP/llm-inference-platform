import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

import httpx

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

# arXiv asks for one request every three seconds; going faster gets you throttled
# rather than banned, but the throttle is slower than just waiting
DELAY_S = 3.0


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


def search(query: str, limit: int, page_size: int = 200, sleep=time.sleep) -> list[Paper]:
    papers: list[Paper] = []
    with httpx.Client(timeout=60) as client:
        for start in range(0, limit, page_size):
            r = client.get(
                API,
                params={
                    "search_query": query,
                    "start": start,
                    "max_results": min(page_size, limit - start),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            r.raise_for_status()
            batch = parse_feed(r.text)
            if not batch:
                break
            papers.extend(batch)
            sleep(DELAY_S)
    return papers
