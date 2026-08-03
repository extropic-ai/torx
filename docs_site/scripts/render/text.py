"""Small text-substitution helpers shared across the renderer."""

import re

# an anchor open tag ending just before a span means the span is already linked;
# 128-char window: the longest anchor tag in the corpus is ~58 chars
_ANCHOR_OPEN_RE = re.compile(r"<a\b[^>]*>\s*$")


def replace_once(text, old, new, label):
    """Substitute the single expected occurrence of ``old`` with ``new``.

    Raises if ``old`` does not appear exactly once, so a template change that
    drops or duplicates a marker fails loudly at build time instead of silently
    emitting a half-substituted page. ``label`` names the marker in the error.
    """
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def span_already_linked(html, start, window=128):
    """True when the span starting at ``start`` sits inside an anchor open tag,
    so a linkifier can skip it and never nest anchors."""
    return bool(_ANCHOR_OPEN_RE.search(html[max(0, start - window) : start]))
