"""Shared site manifest for notebooks, navigation, and API pages."""

import html as html_lib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple


def _find_repo_root(start: Path) -> Path:
    # anchor on `pyproject.toml` rather than a fixed parents[N] depth, so moving
    # this file does not silently repoint ROOT/REPO_ROOT
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"repo root not found above {start} (no pyproject.toml)")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
ROOT = REPO_ROOT / "docs_site"
# gallery notebooks live in the repo-root `examples/` directory (their own PR);
# rendered output and brand media stay under `docs_site/`
NOTEBOOK_DIR = REPO_ROOT / "examples"
OUT_DIR = ROOT / "rendered"
_NOTEBOOK_STEM_RE = re.compile(r"^(?P<number>\d{2})_.+")

# `examples/` also holds torx's pre-existing `basic_usage` quickstart, which
# isn't part of the 16-notebook gallery, so it's excluded from discovery
TORX_OWN_EXAMPLE_STEMS = frozenset({"basic_usage"})

REPO_URL = "https://github.com/extropic-ai/torx"
# `SITE_URL` is the canonical host for absolute doc URLs in llms.txt.
SITE_URL = "https://docs.torx.ai/en/latest"
# Licensed fonts/footer video aren't committed (commercial license); the build
# copies them from the private docs-assets checkout into rendered/ and
# serves them page-relative (./fonts, ./assets) from the docs host -- no CDN.
# All pages live at the output root, so a relative ref resolves on prod and RTD
# preview alike.
ASSET_CDN = "."

# docs-assets checkout; Read the Docs clones it here at build time.
DOCS_ASSETS_DIR = Path(os.environ.get("TORX_DOCS_ASSETS") or (ROOT / "_assets"))

# brand media copied into rendered/assets/ each build; landing and getting-started reference these
BRAND_MEDIA = ("extropic_wordmark.png", "first_circuit.png", "logo.svg")


def repository_source_url(path):
    relative = Path(path).relative_to(REPO_ROOT)
    return f"{REPO_URL}/blob/main/{relative.as_posix()}"


class NotebookEntry(NamedTuple):
    # number/title/stem derive from the filename; blurb/section/featured come
    # from each notebook's ``metadata.docs`` block, so the catalog is the
    # notebooks themselves, not a parallel hand-maintained table
    number: str
    title: str
    stem: str
    blurb: str
    section: str
    featured: bool

    @property
    def href(self):
        return f"{self.stem}.html"

    @property
    def source_url(self):
        return repository_source_url(NOTEBOOK_DIR / f"{self.stem}.ipynb")


class Section(NamedTuple):
    # an editorial section: ordering, heading, blurb, and the notebooks that
    # named its key (in number order); which notebooks land here is read from
    # ``metadata.docs.section`` at build time; the ``SECTIONS`` templates leave
    # ``entries`` empty and ``make_site`` returns copies with it populated
    key: str
    title: str
    blurb: str
    entries: tuple[NotebookEntry, ...] = ()


class ApiCategory(NamedTuple):
    # category membership is derived from each symbol's ``__module__`` at build
    # time, not a hand-maintained name list, so the page tracks the package: a
    # symbol belongs here when its module equals or is a submodule of one of
    # ``module_prefixes``; see ``api_docs._category_assignment``. ``symbols`` is
    # the concrete per-build symbol list, left empty in the curated templates and
    # populated by ``api_docs.api_nav`` (like ``Section.entries`` and ``make_site``)
    label: str
    slug: str
    blurb: str
    module_prefixes: tuple[str, ...]
    symbols: tuple[str, ...] = ()


class Site(NamedTuple):
    # reading_order: `SECTIONS` order then number; drives prev/next nav and is
    #   the single notebook list (derive a number-sorted view where one is needed)
    # sections: editorial sections, each populated with its notebooks (render order)
    # featured: notebooks flagged `docs.featured`, in reading order
    reading_order: tuple[NotebookEntry, ...]
    sections: tuple[Section, ...]
    featured: tuple[NotebookEntry, ...]


def make_site(entries):
    # group notebooks under SECTIONS by their `docs.section` key; the catalog is
    # validated first, so every entry's section resolves to exactly one Section
    entries = sorted(entries, key=lambda entry: int(entry.number))
    groups = []
    reading = []
    for section in SECTIONS:
        members = tuple(
            entry for entry in entries if entry.section == section.key
        )  # entries are already number-sorted
        groups.append(section._replace(entries=members))
        reading.extend(members)
    reading_order = tuple(reading)
    featured = tuple(entry for entry in reading_order if entry.featured)
    return Site(reading_order, tuple(groups), featured)


def notebook_paths():
    return tuple(
        path
        for path in sorted(NOTEBOOK_DIR.glob("[0-9]*.ipynb"))
        if path.stem not in TORX_OWN_EXAMPLE_STEMS
    )


def _docs_from_path(path):
    # validation runs before the build loads notebooks for export, so read the
    # metadata block straight from the file rather than parsing the whole node;
    # this mirrors the build-time ``nb["metadata"]["docs"]`` read on the same block
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("metadata", {}).get("docs")


def validate_notebook_catalog(paths=None):
    paths = notebook_paths() if paths is None else tuple(paths)
    section_keys = {section.key for section in SECTIONS}
    errors = []
    numbers = []
    for path in paths:
        match = _NOTEBOOK_STEM_RE.match(path.stem)
        if match:
            numbers.append(match.group("number"))
        else:
            errors.append(
                f"{path.name}: notebook filename must start with a two-digit number "
                "and underscore"
            )
    if len(set(numbers)) != len(numbers):
        errors.append(f"duplicate notebook numbers: {numbers}")
    section_counts = dict.fromkeys(section_keys, 0)
    for path in paths:
        docs = _docs_from_path(path)
        if docs is None:
            errors.append(f"{path.name}: missing metadata.docs block")
            continue
        if not isinstance(docs, Mapping):
            errors.append(f"{path.name}: metadata.docs must be an object")
            continue
        missing_keys = {"section", "blurb", "featured"} - set(docs)
        if missing_keys:
            errors.append(
                f"{path.name}: docs block missing keys {sorted(missing_keys)}"
            )
        section = docs.get("section")
        if not isinstance(section, str):
            errors.append(f"{path.name}: docs.section must be a string")
        elif section not in section_keys:
            errors.append(
                f"{path.name}: docs.section {section!r} is not a SECTIONS key "
                f"{sorted(section_keys)}"
            )
        else:
            section_counts[section] += 1
        if not isinstance(docs.get("blurb"), str):
            errors.append(f"{path.name}: docs.blurb must be a string")
        if not isinstance(docs.get("featured"), bool):
            errors.append(f"{path.name}: docs.featured must be a bool")
    for key, count in section_counts.items():
        if count == 0:
            errors.append(f"SECTIONS key {key!r} has no notebooks")
    return errors


SECTIONS = [
    Section(
        "foundations",
        "Foundations",
        "The primitives, the core gates, and the circuit-to-simulator pipeline.",
    ),
    Section(
        "discrete-models",
        "Discrete models",
        "Boltzmann machines, diffusion, and trained circuits on pbits.",
    ),
    Section(
        "continuous-hybrid",
        "Continuous and hybrid",
        "Gaussian gates, state-space models, and mixed discrete and continuous processes.",
    ),
    Section(
        "factor-graphs",
        "Directed factor graphs",
        "The layer beneath the circuits: custom factors wired into a directed graph and sampled, with a PSC as one shape it takes.",
    ),
]

# curated API pages; membership is derived from each public symbol's
# ``__module__`` (see ``api_docs._category_assignment``), so adding a gate or
# factor to the package surfaces it on the right page with no edit here; a
# symbol matches when its module equals or is a submodule of a listed prefix
API_CATEGORIES = [
    ApiCategory(
        "Core / factor graphs",
        "api-core",
        "The directed factor graph beneath the circuits: a graph of factors, "
        "each a sampler over its sites.",
        (
            "torx.dfg",
            "torx.factor",
            "torx.composite_factors",
            "torx.tractable_prob_factors",
        ),
    ),
    ApiCategory(
        "Circuits",
        "api-circuits",
        "A circuit is an ordered list of gates applied to an initial state.",
        ("torx.psc._circuit",),
    ),
    ApiCategory(
        "Gates",
        "api-gates",
        "Gates are stochastic kernels acting on pbits, pdits, and pmodes.",
        ("torx.psc.gates",),
    ),
    ApiCategory(
        "Simulators",
        "api-simulators",
        "A simulator compiles a circuit and reads it back as samples, moments, or a density.",
        ("torx.psc.simulation",),
    ),
    ApiCategory(
        "Visualization",
        "api-visualization",
        "Render a circuit as text.",
        ("torx.psc.visualization",),
    ),
]

# Public, non-module symbols intentionally kept off every page.
API_PUBLIC_EXCLUSIONS = frozenset()


DEFAULT_DESCRIPTION = (
    "Torx is a JAX framework for parametrised stochastic circuits: programs that "
    "transform probability distributions, compiled, simulated, and sampled end to end."
)


def og_meta(title, page, description=DEFAULT_DESCRIPTION):
    """Open Graph + Twitter-card <head> tags for social link unfurls.

    URLs canonicalize to the published docs host (SITE_URL); og.png ships into the
    site root. The homepage's canonical URL is the bare host, not /index.html.
    """
    base = SITE_URL.rstrip("/")
    url = f"{base}/" if page == "index.html" else f"{base}/{page}"
    image = f"{base}/og.png"
    title = html_lib.escape(title)
    description = html_lib.escape(description)
    url = html_lib.escape(url)
    image = html_lib.escape(image)
    return (
        '<meta property="og:type" content="website">\n'
        '<meta property="og:site_name" content="Torx">\n'
        f'<meta property="og:title" content="{title}">\n'
        f'<meta property="og:description" content="{description}">\n'
        f'<meta property="og:url" content="{url}">\n'
        f'<meta property="og:image" content="{image}">\n'
        '<meta property="og:image:width" content="1200">\n'
        '<meta property="og:image:height" content="630">\n'
        '<meta property="og:image:alt" content="Torx documentation social preview">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
    )
