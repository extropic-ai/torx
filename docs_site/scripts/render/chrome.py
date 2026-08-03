"""Shared page chrome and navigation helpers."""

import html as html_lib
import re

from pygments import highlight as _pyg_highlight
from pygments.formatters import HtmlFormatter as _PygHtml
from pygments.lexers import get_lexer_by_name, PythonLexer

from .assets import (
    COPY_SCRIPT,
    LOGO_SVG,
    NAV_TRANSITION,
    PAGE_SCRIPT,
    PRELUDE_CSS,
    THEME_CSS,
)
from .manifest import og_meta, REPO_URL
from .text import replace_once


def build_topbar():
    return (
        '<header class="tx-topbar">'
        '<div class="tx-bar-left">'
        '<button class="tx-burger" type="button" aria-label="Open navigation" '
        'aria-expanded="false" aria-controls="tx-sidebar">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>'
        "</button>"
        '<a class="tx-brand" href="index.html">'
        + LOGO_SVG
        + '<span class="tx-title">TORX</span></a>'
        "</div>"
        '<nav class="tx-pills">'
        '<a class="tx-pill" href="examples.html">Examples</a>'
        '<a class="tx-pill" href="' + REPO_URL + '">GitHub</a>'
        "</nav></header>"
    )


def build_sidebar(site, api_categories, active=None):
    # `active` matches one of three disjoint key spaces: doc slug, notebook stem, or API category slug
    parts = ['<aside id="tx-sidebar" class="tx-sidebar"><nav class="tx-sidenav">']

    # styled like the section heads so the top-level entries read as one tier
    gs_cls = (
        "tx-nav-section active" if active == "getting-started" else "tx-nav-section"
    )
    parts.append(f'<a class="{gs_cls}" href="getting-started.html">Getting started</a>')
    examples_cls = "tx-nav-section active" if active == "examples" else "tx-nav-section"
    parts.append(f'<a class="{examples_cls}" href="examples.html">Examples</a>')
    for section in site.sections:
        parts.append(
            f'<div class="tx-nav-group"><div class="tx-nav-grouphead">{html_lib.escape(section.title, quote=False)}</div>'
        )
        for entry in section.entries:
            cls = "tx-nav-nb active" if active == entry.stem else "tx-nav-nb"
            parts.append(
                f'<a class="{cls}" href="{entry.href}"><span class="tx-nav-num">{entry.number}</span> <span class="tx-nav-title">{html_lib.escape(entry.title, quote=False)}</span></a>'
            )
        parts.append("</div>")
    parts.append('<div class="tx-nav-section">API reference</div>')
    parts.append('<div class="tx-nav-group">')
    for category in api_categories:
        cls = "tx-nav-link active" if active == category.slug else "tx-nav-link"
        parts.append(
            f'<a class="{cls}" href="{category.slug}.html">{html_lib.escape(category.label, quote=False)}</a>'
        )
    parts.append("</div>")
    parts.append("</nav></aside>")
    return "".join(parts)


def highlight_tokens(code, lang="python"):
    """Pygments-highlight ``code`` into bare ``.highlight`` token spans.

    Shared by every code surface (doc cards, the landing quickstart) so all of
    them pick up the palette in prelude.css.
    """
    try:
        lexer = PythonLexer() if lang == "python" else get_lexer_by_name(lang)
    except Exception:
        lexer = PythonLexer()
    return _pyg_highlight(code.strip("\n"), lexer, _PygHtml(nowrap=True)).rstrip("\n")


def code_card(code, lang="python"):
    """A self-contained docs code card matching the notebook code-cell look."""
    tokens = highlight_tokens(code, lang)
    return (
        '<div class="tx-codecard">'
        '<div class="tx-code-head"><span class="tx-lang"></span>'
        '<button class="tx-copy" type="button" title="Copy code" aria-label="Copy code"></button></div>'
        '<pre><code class="highlight">' + tokens + "</code></pre>"
        "</div>"
    )


def _inject_body(html, body_class, chrome):
    """Add a body class and inject the chrome markup at the real <body> tag."""
    matches = list(re.finditer(r"<body([^>]*)>", html))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one <body> tag, found {len(matches)}")
    match = matches[0]
    attrs = match.group(1)
    if 'class="' in attrs:
        attrs = re.sub(r'class="', f'class="{body_class} ', attrs, count=1)
    else:
        attrs = attrs + f' class="{body_class}"'
    return html[: match.start()] + f"<body{attrs}>{chrome}" + html[match.end() :]


def build_notebook_actions(entry):
    """Link a rendered notebook back to its source."""
    source_url = html_lib.escape(entry.source_url, quote=True)
    return (
        '<nav class="tx-notebook-actions" aria-label="Notebook resources">'
        f'<a href="{source_url}">View source</a>'
        "</nav>"
    )


def prev_next_nav(site, active_stem):
    """Previous / next notebook links for the bottom of a notebook page."""
    ordered = site.reading_order
    idx = next(
        (i for i, entry in enumerate(ordered) if entry.stem == active_stem), None
    )
    if idx is None:
        raise RuntimeError(f"active notebook {active_stem!r} not in reading order")

    def card(entry, direction):
        arrow = "&larr; Previous" if direction == "prev" else "Next &rarr;"
        return (
            f'<a class="tx-pn tx-pn-{direction}" href="{entry.href}">'
            f'<span class="tx-pn-dir">{arrow}</span>'
            f'<span class="tx-pn-title"><span class="tx-pn-num">{entry.number}</span> '
            f"{html_lib.escape(entry.title)}</span></a>"
        )

    prev_html = (
        card(ordered[idx - 1], "prev")
        if idx > 0
        else '<span class="tx-pn tx-pn-empty"></span>'
    )
    next_html = (
        card(ordered[idx + 1], "next")
        if idx < len(ordered) - 1
        else '<span class="tx-pn tx-pn-empty"></span>'
    )
    return (
        '<nav class="tx-pagenav" aria-label="Notebook navigation">'
        + prev_html
        + next_html
        + "</nav>"
    )


def _inject_after_content_main(html, insertion):
    match = re.search(r"</aside>\s*<main(?:\s[^>]*)?>", html)
    if match is None:
        raise RuntimeError("missing content <main> marker")
    return html[: match.end()] + insertion + html[match.end() :]


def inject_chrome(html, site, api_categories, active_stem, title):
    """Add the theme, top bar, sidebar, and social-card metadata to a notebook page."""
    head_extra = (
        og_meta(f"{title} - Torx", f"{active_stem}.html")
        + PRELUDE_CSS
        + THEME_CSS
        + NAV_TRANSITION
    )
    html = replace_once(
        html,
        "</head>",
        head_extra + "</head>",
        "</head> marker",
    )
    chrome = build_topbar() + build_sidebar(site, api_categories, active=active_stem)
    html = _inject_body(html, "tx-has-sidebar", chrome)
    entry = next(entry for entry in site.reading_order if entry.stem == active_stem)
    html = _inject_after_content_main(html, build_notebook_actions(entry))
    nav = prev_next_nav(site, active_stem)
    # Anchor to the document's closing body tag, not a cell's raw-HTML output.
    return _insert_before_last(html, "</body>", nav + COPY_SCRIPT + PAGE_SCRIPT)


def _insert_before_last(html, tag, insertion):
    """Insert ``insertion`` before the last occurrence of ``tag``."""
    idx = html.rfind(tag)
    if idx == -1:
        raise RuntimeError(f"missing {tag} marker")
    return html[:idx] + insertion + html[idx:]
