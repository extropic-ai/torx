"""Hand-authored docs pages, landing page, and llms.txt."""

import html as html_lib
import re

from .assets import (
    COPY_SCRIPT,
    DOC_CSS,
    FAVICON_LINK,
    INDEX_CSS,
    INDEX_SCRIPT,
    LOGO_SVG,
    NAV_TRANSITION,
    PAGE_SCRIPT,
    PRELUDE_CSS,
    THEME_CSS,
)
from .chrome import build_sidebar, build_topbar, code_card, highlight_tokens
from .manifest import (
    ASSET_CDN,
    og_meta,
    REPO_ROOT,
    REPO_URL,
    SITE_URL,
)

TORX_INSTALL = "pip install extro-torx"
UV_TORX_INSTALL = "uv pip install extro-torx"
_FIRST_CIRCUIT_OUTPUT = "stay |10): 0.699, swap |01): 0.301"


def _readme_bibtex():
    """The citation block is authored once, in README.md; the site reads it out."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"```bibtex\n(.*?)\n```", readme, re.S)
    if not m:
        raise RuntimeError("README.md no longer contains a bibtex citation block")
    return m.group(1)


FIRST_CIRCUIT = (
    "import jax\n"
    "import jax.numpy as jnp\n"
    "from torx.psc import DiscretePCircuit, BranchingSimulator, PSWAP\n"
    "\n"
    "# PSWAP swaps two pbits with probability sigma(theta); here p(swap) = 0.3.\n"
    "# Gates are structure-only; parameters are a separate list of logit leaves.\n"
    "circuit = DiscretePCircuit([PSWAP([0, 1])])\n"
    "thetas = [jnp.array([jnp.log(0.3 / 0.7)])]\n"
    "\n"
    "sim = BranchingSimulator(num_samples=20_000)\n"
    "compiled = sim.build_circuit(circuit, thetas)\n"
    "\n"
    "# Start in |10) and sample the two output pbits.\n"
    "state10 = jnp.array([1, 0], dtype=jnp.int32)\n"
    "state01 = jnp.array([0, 1], dtype=jnp.int32)\n"
    "samples = sim.sample(compiled, state10, jax.random.key(0))\n"
    "stay = jnp.mean(jnp.all(samples == state10, axis=1))\n"
    "swap = jnp.mean(jnp.all(samples == state01, axis=1))\n"
    'print(f"stay |10): {float(stay):.3f}, swap |01): {float(swap):.3f}")\n'
)

COPY_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="13" height="13" rx="2"/>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
)

MATHJAX = (
    "<script>window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }, "
    "svg: { fontCache: 'global' } };</script>\n"
    '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>\n'
)


def write_doc_page(
    slug,
    title,
    inner,
    site,
    api_categories,
    active_page,
    *,
    mathjax=False,
    active_api=None,
    out_dir,
):
    head = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title} - Torx</title>\n"
        + FAVICON_LINK
        + og_meta(f"{title} - Torx", f"{slug}.html")
        + PRELUDE_CSS
        + THEME_CSS
        + DOC_CSS
        + NAV_TRANSITION
        + (MATHJAX if mathjax else "")
        + "</head>\n"
    )
    body = (
        '<body class="tx-has-sidebar">'
        + build_topbar()
        + build_sidebar(site, api_categories, active=active_api or active_page)
        + f'<main class="tx-doc">{inner}</main>'
        + COPY_SCRIPT
        + PAGE_SCRIPT
        + "</body>\n</html>\n"
    )
    (out_dir / f"{slug}.html").write_text(head + body, encoding="utf-8")


def examples_inner(site):
    """A Browse Examples gallery: every notebook as a card, grouped by section."""
    n_examples = len(site.reading_order)
    parts = [
        "<h1>Examples</h1>",
        f'<p class="lede">The gallery holds {n_examples} notebooks. They start with a '
        "one-gate circuit and work up to trained networks and directed factor graphs. "
        "Each page is a full run of its notebook, with the outputs it produced.</p>",
    ]
    for section in site.sections:
        parts.append(f"<h2>{html_lib.escape(section.title, quote=False)}</h2>")
        if section.blurb:
            parts.append(f"<p>{html_lib.escape(section.blurb, quote=False)}</p>")
        parts.append('<div class="tx-cards">')
        for entry in section.entries:
            parts.append(
                f'<a class="tx-navcard" href="{html_lib.escape(entry.href, quote=True)}">'
                f'<span class="tx-navcard-title">{entry.number} '
                f"{html_lib.escape(entry.title, quote=False)}</span>"
                f'<span class="tx-navcard-blurb">{html_lib.escape(entry.blurb, quote=False)}</span></a>'
            )
        parts.append("</div>")
    return "\n".join(parts)


def getting_started_inner(site):
    intro_href = site.reading_order[0].href
    return (
        "<h1>Getting started</h1>\n"
        '<p class="lede">Torx is a JAX framework for parametrised stochastic circuits (PSCs): '
        "programs that transform probability distributions instead of fixed values. This page installs "
        "Torx and runs a first circuit.</p>\n"
        "<h2>Installation</h2>\n"
        "<p>Torx requires Python 3.11 or newer. Install it from PyPI:</p>\n"
        + code_card(TORX_INSTALL, "bash")
        + '<p>Or with <a href="https://docs.astral.sh/uv/">uv</a>:</p>\n'
        + code_card(UV_TORX_INSTALL, "bash")
        + "<p>If you are developing Torx from a local checkout, install that checkout with the optional test and example dependencies:</p>\n"
        + code_card(
            "git clone https://github.com/extropic-ai/torx.git\n"
            "cd torx\n"
            'pip install -e ".[testing,examples]"',
            "bash",
        )
        + "<h2>Your first circuit</h2>\n"
        "<p>A circuit is an ordered list of gates applied to an initial state. Build a one-gate "
        "circuit from <code>PSWAP</code>, compile it on a <code>BranchingSimulator</code>, and draw samples:</p>\n"
        + code_card(FIRST_CIRCUIT)
        + "<p>Starting from <code>|10)</code>, the swap fires on about 30% of samples, so roughly "
        "70% of outputs stay <code>|10)</code> and 30% become <code>|01)</code>. For this discrete "
        "gate, the transition matrix is column-stochastic, and the simulator pushes samples through it.</p>\n"
        '<figure class="tx-fig"><img src="assets/first_circuit.png" alt="Output distribution from |10): about 70% |10) and 30% |01)"></figure>\n'
        '<div class="note"><span class="note-t">Note</span><p>For this discrete <code>PSWAP</code>, '
        "<code>theta</code> is the logit of the swap probability "
        "<code>p</code>. Apply the sigmoid, "
        "<code>p = sigma(theta)</code>, to read the switching probability back; the logit ranges over all "
        "of the reals, so it trains cleanly with gradients.</p></div>\n"
        "<h2>Running the notebooks</h2>\n"
        "<p>The example notebooks live in the <code>examples/</code> directory of the Torx "
        "repository, alongside their helpers. Clone it, install the example dependencies, and "
        "open the notebooks:</p>\n"
        + code_card(
            "git clone https://github.com/extropic-ai/torx.git\n"
            "cd torx\n"
            'pip install -e ".[examples]" jupyterlab\n'
            "jupyter lab examples/",
            "bash",
        )
        + "<h2>Where to go next</h2>\n"
        '<div class="tx-cards">'
        f'<a class="tx-navcard" href="{intro_href}"><span class="tx-navcard-title">Introduction notebook &rarr;</span>'
        '<p class="tx-navcard-blurb">Build a PSC from scratch and meet the core gate for each of the three primitives.</p></a>'
        "</div>\n" + "<h2>Citation</h2>\n"
        "<p>If you found this library useful in academic research, please cite:</p>\n"
        + code_card(_readme_bibtex(), "bibtex")
        + "<h2>See also</h2>\n"
        "<p>Other libraries in the JAX ecosystem: "
        '<a href="https://github.com/lockwo/awesome-jax">Awesome JAX</a>, '
        "a longer list of other JAX projects.</p>\n"
    )


def _landing_code_card():
    copy_btn = (
        '<button class="tx-copy" type="button" aria-label="Copy code">'
        + COPY_SVG
        + "</button>"
    )
    return (
        '<div class="tx-codecard"><div class="tx-code-head"><span class="tx-lang"></span>'
        + copy_btn
        + '</div><pre><code class="highlight">'
        + highlight_tokens(FIRST_CIRCUIT)
        + "</code></pre>"
        + '<div class="tx-code-output"><span>expected output</span><code>'
        + html_lib.escape(_FIRST_CIRCUIT_OUTPUT, quote=False)
        + "</code></div></div>"
    )


def write_index(site, *, out_dir):
    def card(entry):
        return (
            f'      <a class="tx-card" href="{html_lib.escape(entry.href, quote=True)}">'
            f'<span class="tx-card-num">{entry.number}</span>'
            f'<span class="tx-card-title">{html_lib.escape(entry.title, quote=False)}</span>'
            f'<span class="tx-card-blurb">{html_lib.escape(entry.blurb, quote=False)}</span></a>'
        )

    featured_cards = "\n".join(card(entry) for entry in site.featured)
    browse_href = "examples.html"
    n_examples = len(site.reading_order)
    first_circuit_card = _landing_code_card()

    head = (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Torx - Parametrised Stochastic Circuits</title>\n"
        + FAVICON_LINK
        + '<meta name="description" content="Torx is a JAX framework for parametrised stochastic circuits: programs that transform probability distributions, with docs and runnable example notebooks.">\n'
        + og_meta("Torx - Parametrised Stochastic Circuits", "index.html")
        + PRELUDE_CSS
        + "<style>\n"
        + INDEX_CSS
        + "\n</style>\n"
        + NAV_TRANSITION
        + "</head>\n"
    )
    body = f"""<body>
  <header class="tx-nav">
    <div class="tx-nav-inner">
      <a class="tx-brand" href="index.html">{LOGO_SVG}<span class="tx-brand-name">TORX</span></a>
      <nav class="tx-pills">
        <a class="tx-pill" href="getting-started.html">Docs</a>
        <a class="tx-pill" href="{REPO_URL}">GitHub</a>
      </nav>
    </div>
  </header>

  <section class="tx-hero">
    <div class="tx-hero-text">
      <div class="tx-hero-eyebrow"><img src="assets/extropic_wordmark.png" alt="Extropic"></div>
      <h1>Parametrised stochastic circuits</h1>
      <p class="tx-tagline">Programs that transform probability distributions instead of fixed values. Build circuits
      from gates on three primitives, then sample them or propagate their moments exactly.</p>
      <div class="tx-cta">
        <a class="tx-pill tx-pill-solid" href="getting-started.html">Get started &rarr;</a>
        <a class="tx-pill" href="{REPO_URL}">View on GitHub</a>
      </div>
    </div>
    <div class="tx-hero-visual">
      <div class="tx-led-panel"><canvas id="tx-led" class="tx-led-canvas"></canvas></div>
    </div>
  </section>

  <div class="tx-prims tx-reveal">
    <div class="tx-prim"><span class="tx-prim-name">pbit</span><span class="tx-prim-dom">{{0, 1}}</span><div class="tx-prim-desc">A binary site, flipped and swapped by the discrete gates.</div></div>
    <div class="tx-prim"><span class="tx-prim-name">pdit</span><span class="tx-prim-dom">{{0, &hellip;, d-1}}</span><div class="tx-prim-desc">A d-state site, permuted and cycled by its gates.</div></div>
    <div class="tx-prim"><span class="tx-prim-name">pmode</span><span class="tx-prim-dom">&#8477;<sup>N</sup></span><div class="tx-prim-desc">A continuous site carried by Gaussian gates.</div></div>
  </div>

  <section class="tx-quick tx-reveal">
    <h2 class="tx-sec-title">Quickstart</h2>
    <p class="tx-install">Install with <code>{html_lib.escape(TORX_INSTALL, quote=False)}</code> (Python 3.11+), then build and sample a circuit:</p>
    {first_circuit_card}
  </section>

  <section id="notebooks" class="tx-nb-wrap tx-reveal">
    <h2 class="tx-sec-title">Example notebooks</h2>
    <p class="tx-sec-sub">{n_examples} runnable notebooks, starting from a one-gate circuit and working up to trained networks and directed factor graphs.</p>
    <div class="tx-grid">
{featured_cards}
    </div>
    <a class="tx-browse" href="{browse_href}">Browse all {n_examples} examples &rarr;</a>
  </section>

  <footer class="tx-foot">
    <video class="tx-foot-video" autoplay loop muted playsinline>
      <source src="{ASSET_CDN}/assets/extropic-footer.mp4" type="video/mp4">
    </video>
    <div class="tx-foot-inner">
      <span>TORX <a href="https://extropic.ai/">EXTROPIC</a></span>
      <span><a href="{REPO_URL}">GitHub</a></span>
    </div>
  </footer>
{COPY_SCRIPT}
{INDEX_SCRIPT}
</body>
</html>
"""
    (out_dir / "index.html").write_text(head + body, encoding="utf-8")


def write_llms_txt(site, api_categories, *, out_dir):
    """Write an llms.txt index of the docs, examples, and API for AI consumption."""
    out = [
        "# Torx",
        "",
        "> Torx is a JAX framework for parametrised stochastic circuits (PSCs): programs that transform "
        "probability distributions instead of fixed values. Circuits are built from gates acting on three "
        "data primitives, the pbit (binary), pdit (d-state), and pmode (continuous), then read back by a "
        "simulator as samples, exact moments, or an exact density. The API reference below covers the "
        "torx package installed for this build.",
        "",
        "## Start here",
        f"- [Documentation home]({SITE_URL}/index.html): the Torx guides, examples, and API reference.",
        f"- [Getting started]({SITE_URL}/getting-started.html): install Torx and build and sample a first circuit.",
        f"- [Examples index]({SITE_URL}/examples.html): all runnable notebooks in learning order.",
        f"- [API reference]({SITE_URL}/api-core.html): start with the core factor-graph API, then follow the category links.",
        f"- [GitHub repository]({REPO_URL}): package and notebook source.",
        "",
        "## Examples",
    ]
    # listed in notebook-number order (a sorted view of the single reading_order)
    for entry in sorted(site.reading_order, key=lambda entry: entry.number):
        out.append(
            f"- [{entry.number} {entry.title}]({SITE_URL}/{entry.href}): {entry.blurb}"
        )
    out += ["", "## API reference"]
    for category in api_categories:
        present = ", ".join(category.symbols)
        out.append(
            f"- [{category.label}]({SITE_URL}/{category.slug}.html): {category.blurb} ({present})"
        )
    (out_dir / "llms.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


LEGACY_REDIRECTS = {
    "api/dfg/index.html": "../../api-core.html#DFG",
    "api/factors/index.html": "../../api-core.html",
    "api/gates/index.html": "../../api-gates.html",
    "api/circuit/index.html": "../../api-circuits.html",
    "api/composite/index.html": "../../api-core.html",
    "api/simulators/index.html": "../../api-simulators.html",
    "api/visualization/index.html": "../../api-visualization.html",
}


def write_legacy_redirects(*, out_dir):
    for path, target in LEGACY_REDIRECTS.items():
        dest = out_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        safe_target = html_lib.escape(target, quote=True)
        target_text = html_lib.escape(target, quote=False)
        dest.write_text(
            "<!doctype html>\n"
            '<html lang="en"><head><meta charset="utf-8">\n'
            f'<meta http-equiv="refresh" content="0; url={safe_target}">\n'
            f'<link rel="canonical" href="{safe_target}">\n'
            f"<title>Redirecting to {target_text}</title>\n"
            "</head><body>"
            f'<p>Redirecting to <a href="{safe_target}">{target_text}</a>.</p>'
            "</body></html>\n",
            encoding="utf-8",
        )
