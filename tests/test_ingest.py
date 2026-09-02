from datetime import datetime

from app.arxiv import parse_feed
from app.chunking import chunk

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v3</id>
    <updated>2024-02-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Paged Attention
      for Serving</title>
    <summary>  We reduce   KV cache
      fragmentation.  </summary>
    <category term="cs.LG"/>
    <category term="cs.DC"/>
  </entry>
</feed>"""


def test_parse_feed_strips_version_and_whitespace():
    (p,) = parse_feed(FEED)

    # the version suffix would fragment the primary key across re-ingests
    assert p.arxiv_id == "2401.00001"
    assert p.title == "Paged Attention for Serving"
    assert p.abstract == "We reduce KV cache fragmentation."
    assert p.categories == ["cs.LG", "cs.DC"]
    assert p.published == datetime.fromisoformat("2024-01-01T00:00:00Z")
    assert p.updated == datetime.fromisoformat("2024-02-01T00:00:00Z")


def test_parse_feed_ignores_entries_without_id():
    assert parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"><entry/></feed>') == []


def test_short_text_is_one_chunk():
    assert chunk("  a  short   abstract ") == ["a short abstract"]


def test_empty_text_yields_nothing():
    assert chunk("   ") == []


def test_long_text_splits_with_overlap_on_word_boundaries():
    words = " ".join(f"word{i}" for i in range(900))
    parts = chunk(words, max_chars=500, overlap=100)

    assert len(parts) > 1
    assert all(len(p) <= 500 for p in parts)
    assert not any(p.startswith(" ") or p.endswith(" ") for p in parts)
    # overlap means the tail of one chunk reappears at the head of the next
    assert parts[0].split()[-1] in parts[1].split()
    # every word survives somewhere
    assert set(words.split()) == {w for p in parts for w in p.split()}


def test_overlap_never_shrinks_when_snapping_to_a_word():
    words = " ".join(f"{'w' * 9}{i}" for i in range(2000))
    parts = chunk(words, max_chars=1800, overlap=240)

    starts, cur = [], 0
    for p in parts:
        i = words.index(p, cur)
        starts.append(i)
        cur = i + 1
    widths = [starts[i] + len(parts[i]) - starts[i + 1] for i in range(len(parts) - 1)]

    longest = max(len(w) for w in words.split())
    # snapping moves the resume point backward to a word start, so the real
    # overlap is never below target — it runs over by at most one word
    assert min(widths) >= 240
    assert max(widths) <= 240 + longest + 1


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split())[:60], params))

    def executemany(self, sql, params=None):
        self.log.append(("executemany", " ".join(sql.split())[:60], params))


class FakeConn:
    def __init__(self):
        self.log = []

    def cursor(self):
        return FakeCursor(self.log)

    def commit(self):
        self.log.append(("commit", "", None))


def test_store_page_clears_old_chunks_before_writing():
    from datetime import UTC, datetime

    from app.arxiv import Paper
    from app.ingest import store_page

    conn = FakeConn()
    paper = Paper("2401.00001", "t", "a", ["cs.LG"], datetime.now(UTC), None)

    n = store_page(conn, [paper])

    verbs = [(kind, sql.split()[0]) for kind, sql, _ in conn.log if kind != "commit"]
    # the delete has to land between the papers upsert and the chunk insert, or a
    # re-chunk leaves orphaned high-ord rows the upsert never touches
    assert verbs == [
        ("executemany", "insert"),
        ("execute", "delete"),
        ("executemany", "insert"),
    ]
    assert n == 1
