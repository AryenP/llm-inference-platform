# Character budget rather than tokens: pulling in a tokenizer to split text this
# short would cost more than it buys.
#
# 1800 measured against the real corpus: 8,009 of 10,000 papers land in one
# chunk and 1,991 in two, never more. The second chunk is never a stub — the
# overlap floor makes it at least OVERLAP_CHARS long. Granularity is therefore
# close to one abstract per vector, which is a property of a corpus of abstracts
# rather than a tuning choice.
MAX_CHARS = 1800
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
