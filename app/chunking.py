# Character budget rather than tokens: pulling in a tokenizer to split text this
# short would cost more than it buys.
#
# 2400 is chosen so an arXiv title+abstract is always exactly one chunk — arXiv
# caps abstracts near 1920 characters, and a 1800 budget split the longest few
# into a full chunk plus a ~120-character stub, which would sit in the index as a
# near-contentless vector. Retrieval granularity here is deliberately one
# abstract per vector. The splitting path exists for the full-text upgrade.
MAX_CHARS = 2400
OVERLAP_CHARS = 240


def chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []

    out, start = [], 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # break on a space so a chunk never ends mid-word
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        # resume from a word boundary back from the cut, so the overlap doesn't
        # hand the next chunk a fragment like "d210" instead of "word210"
        resume = max(end - overlap, start + 1)
        space = text.rfind(" ", start, resume)
        start = space + 1 if space > start else resume
    return [c for c in out if c]
