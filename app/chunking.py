# Character budget rather than tokens: abstracts are short enough that nearly all
# land in one chunk, and pulling in a tokenizer to split the handful that don't
# would cost more than it buys.
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
