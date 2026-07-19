"""Notebook export helpers."""

import base64
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

from .manifest import OUT_DIR, REPO_ROOT, REPO_URL

NOTEBOOK_FIG_DIR = "notebooks"
_IMG_TAG_RE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE)
_SRC_DATA_RE = re.compile(r'src="data:image/(png|jpeg|gif);base64,([^"]+)"')
# jpeg's conventional file extension is .jpg; png/gif match their subtype
_IMG_EXT = {"png": "png", "jpeg": "jpg", "gif": "gif"}
# case-insensitive to match the smoke check, so a `.IPYNB` link is rewritten, not flagged
_HREF_IPYNB_RE = re.compile(r'href="([^"]*?\.ipynb(?:\?[^"#]*)?(?:#[^"]*)?)"', re.I)
# Prose codespans naming a shipped helper module become links to the file on GitHub,
# so "defined in examples/helpers/_langevin.py" is one click away from the source.
_HELPER_CODE_RE = re.compile(r"<code>(examples/helpers/_[a-z0-9_]+\.py)</code>")
# lookbehind window for an already-linked code span, mirroring api_docs.linkify_api
_HELPER_LINK_PREFIX_WINDOW = 128


def linkify_helper_paths(html):
    """Wrap ``examples/helpers/_*.py`` codespans in links to the file on GitHub.

    Paths are checked against the working tree; a mention of a helper that does
    not exist is a docs bug and fails the build. A code span already inside an
    anchor is skipped so links never nest (same guard as ``linkify_api``).
    """

    def repl(m):
        prefix = html[max(0, m.start() - _HELPER_LINK_PREFIX_WINDOW) : m.start()]
        if re.search(r"<a\b[^>]*>\s*$", prefix):
            return m.group(0)
        rel = m.group(1)
        if not (REPO_ROOT / rel).is_file():
            raise RuntimeError(f"prose names a missing helper module: {rel}")
        return f'<a href="{REPO_URL}/blob/main/{rel}">{m.group(0)}</a>'

    return _HELPER_CODE_RE.sub(repl, html)


def tag_hidden_inputs(nb):
    """Mirror each cell's JupyterLab source_hidden state onto a hide-input tag.

    Mutates ``nb`` in place. ``nb`` is the freshly read in-memory notebook used
    only for export, so the source ``.ipynb`` on disk is never touched.
    """
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        hidden = cell.get("metadata", {}).get("jupyter", {}).get("source_hidden", False)
        if not hidden:
            continue
        tags = list(cell.get("metadata", {}).get("tags", []))
        if "hide-input" not in tags:
            tags.append("hide-input")
        cell.metadata["tags"] = tags


def _png_size(raw):
    """Return (width, height) from a PNG's IHDR, or None if not a PNG."""
    # width/height live at IHDR bytes 16-24, after the 8-byte signature
    if len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return None


def _fig_dir(out_dir):
    return out_dir / "assets" / NOTEBOOK_FIG_DIR


def externalize_images(body, stem, out_dir=OUT_DIR):
    """Write inline base64 figure PNGs out to assets/notebooks/ and rewrite the
    <img> tags to reference them, with lazy/async loading and intrinsic height.

    nbconvert's lab template inlines every output figure as a base64 data URI,
    which the browser must download and parse before first paint. Externalizing
    the figures lets the browser cache and parallel-load images. The lab-template
    favicon lives in CSS as url(data:...), so it is left untouched.
    """
    fig_dir = _fig_dir(out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def repl(match):
        tag = match.group(0)
        data = _SRC_DATA_RE.search(tag)
        if not data:
            # unsupported subtype or external `<img>` (no inline base64): pass through
            return tag
        ext = _IMG_EXT[data.group(1)]
        raw = base64.b64decode(data.group(2))
        digest = hashlib.sha1(raw).hexdigest()[:12]
        fname = f"{stem}_{digest}.{ext}"
        (fig_dir / fname).write_bytes(raw)
        tag = _SRC_DATA_RE.sub(f'src="assets/{NOTEBOOK_FIG_DIR}/{fname}"', tag, count=1)

        extra = ""
        if "loading=" not in tag:
            extra += ' loading="lazy" decoding="async"'
        size = _png_size(raw)
        width_attr = re.search(r'width="(\d+)"', tag)
        if size and width_attr and "height=" not in tag:
            disp_w = int(width_attr.group(1))
            png_w, png_h = size
            if png_w:
                extra += f' height="{round(disp_w * png_h / png_w)}"'
        if extra:
            if tag.endswith("/>"):
                tag = tag[:-2] + extra + "/>"
            elif tag.endswith(">"):
                tag = tag[:-1] + extra + ">"
        return tag

    return _IMG_TAG_RE.sub(repl, body)


def prune_unreferenced_pngs(out_dir=OUT_DIR):
    fig_dir = _fig_dir(out_dir)
    if not fig_dir.exists():
        return []
    refs = set()
    for html_path in out_dir.rglob("*.html"):
        refs.update(
            re.findall(
                rf"assets/{NOTEBOOK_FIG_DIR}/([^\"')]+\.(?:png|jpg|gif))",
                html_path.read_text(encoding="utf-8"),
            )
        )
    removed = []
    for path in sorted(fig_dir.iterdir()):
        if path.is_file() and path.name not in refs:
            path.unlink()
            removed.append(path)
    return removed


def normalize_cell_ids(body):
    """Strip nbconvert's nondeterministic ``cell-id`` attributes for reproducible builds."""
    return re.sub(r' id="cell-id=[^"]+"', "", body)


def notebook_title(nb, fallback):
    for cell in nb.cells:
        if cell.get("cell_type") == "markdown":
            match = re.search(
                r"^#\s+(.+)", "".join(cell.get("source", "")), re.MULTILINE
            )
            if match:
                return match.group(1).strip()
    return fallback


def rewrite_nb_links(html):
    """Rewrite relative cross-notebook links from .ipynb to .html."""
    return _HREF_IPYNB_RE.sub(_nb_to_html, html)


def _nb_to_html(m):
    href = m.group(1)
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or not parsed.path.lower().endswith(".ipynb"):
        return m.group(0)
    path = parsed.path[:-6] + ".html"
    return f'href="{urlunsplit(("", "", path, parsed.query, parsed.fragment))}"'
