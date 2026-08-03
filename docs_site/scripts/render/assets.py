"""Static site assets loaded and inlined during rendering.

Most CSS/JS assets self-wrap in ``<style>``/``<script>`` tags, so consumers can
concatenate them directly. ``index.css`` stays pure CSS and is wrapped by the
landing-page template.
"""

from .manifest import ASSET_CDN, ROOT

# `ROOT` is the `docs_site/` dir resolved from the repo's `pyproject.toml` anchor
ASSET_DIR = ROOT / "scripts" / "site_assets"

# runtime contract: rendered docs reference fonts page-relative (./fonts/*) and
# the footer video at ./assets/extropic-footer.mp4; both copied in at build time
_ASSET_CDN_TOKEN = "__ASSET_CDN__"


def read_site_asset(name):
    text = (ASSET_DIR / name).read_text(encoding="utf-8")
    rendered = text.replace(_ASSET_CDN_TOKEN + "/", ASSET_CDN.rstrip("/") + "/")
    if _ASSET_CDN_TOKEN in rendered:
        raise RuntimeError(
            f"asset {name!r} still contains the {_ASSET_CDN_TOKEN} sentinel after "
            "substitution; check the asset host placeholder"
        )
    return rendered


PRELUDE_CSS = read_site_asset("prelude.css")
THEME_CSS = read_site_asset("theme.css")
DOC_CSS = read_site_asset("doc.css")
INDEX_CSS = read_site_asset("index.css")
COPY_SCRIPT = read_site_asset("copy.js")
PAGE_SCRIPT = read_site_asset("page.js")
INDEX_SCRIPT = read_site_asset("index.js")
NAV_TRANSITION = read_site_asset("nav_transition.html")
# torx mark inlined into the top bar and landing header
LOGO_SVG = read_site_asset("logo.svg").strip()
