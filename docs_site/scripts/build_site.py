"""Render the Torx docs site.

The site has four kinds of page:

* ``index.html``: a full-bleed landing page.
* ``getting-started.html`` and ``examples.html``: hand-authored prose docs.
* ``api-*.html``: API reference pages generated from the installed torx package.
* ``NN_name.html``: example notebooks exported from ``.ipynb`` with shared chrome.

GitHub ignores ``metadata.jupyter.source_hidden``; these renders honor it, so a
cell collapsed in JupyterLab folds behind a gutter chevron here and the page
opens results-forward. PNG output figures are extracted to
``rendered/assets/notebooks/``.

Usage:
    uv run --extra docs python docs_site/scripts/build_site.py

The source notebooks are never modified: the hide-input tagging happens on an
in-memory copy used only for export.
"""

import html as html_lib
import os
import re
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from render.api_docs import api_inner, api_nav, linkify_api, validate_api_reference
from render.chrome import inject_chrome
from render.manifest import (
    BRAND_MEDIA,
    DOCS_ASSETS_DIR,
    make_site,
    notebook_paths,
    NotebookEntry,
    OUT_DIR,
    REPO_ROOT,
    ROOT,
    validate_notebook_catalog,
)
from render.notebooks import (
    externalize_images,
    linkify_helper_paths,
    normalize_cell_ids,
    notebook_title,
    prune_unreferenced_pngs,
    rewrite_nb_links,
    tag_hidden_inputs,
    validate_source_excerpts,
)
from render.pages import (
    examples_inner,
    getting_started_inner,
    LEGACY_REDIRECTS,
    write_doc_page,
    write_index,
    write_legacy_redirects,
    write_llms_txt,
)
from render.text import replace_once


def _fail_if_errors(errors, label):
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"{label} failed:\n{joined}")


_RENDERED_GITIGNORE = "*\n!.gitignore\n"


def _running_on_readthedocs():
    return os.environ.get("READTHEDOCS", "").lower() == "true"


def _required_font_names():
    css = (ROOT / "scripts" / "site_assets" / "prelude.css").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"__ASSET_CDN__/fonts/([^)'\"\s]+)", css)))


def copy_brand_media(out_dir):
    """Copy the committed brand media into the build's assets directory.

    The landing and getting-started pages reference these images by their
    ``assets/`` path; the licensed fonts/footer video are copied separately by
    ``copy_licensed_assets``.
    """
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    brand_dir = ROOT / "assets" / "brand"
    for name in BRAND_MEDIA:
        shutil.copy2(brand_dir / name, assets_dir / name)
    # Copy the social-card image at the site root so og:image resolves to /og.png.
    shutil.copy2(brand_dir / "og.png", out_dir / "og.png")
    copy_licensed_assets(out_dir)


def copy_licensed_assets(out_dir):
    """Copy the licensed fonts + footer video from the docs-assets checkout
    into the build, so they're served from the docs host (./fonts, ./assets) with
    no CDN. Absent checkout (local/CI) -> skip; pages fall back to system fonts."""
    missing = []
    fonts_src = DOCS_ASSETS_DIR / "fonts"
    if fonts_src.is_dir():
        for name in _required_font_names():
            if not (fonts_src / name).is_file():
                missing.append(fonts_src / name)
        fonts_dst = out_dir / "fonts"
        fonts_dst.mkdir(parents=True, exist_ok=True)
        for src in sorted(fonts_src.iterdir()):
            if src.is_file():
                shutil.copy2(src, fonts_dst / src.name)
    else:
        missing.append(fonts_src)
    footer = DOCS_ASSETS_DIR / "videos" / "extropic-footer.mp4"
    if footer.is_file():
        shutil.copy2(footer, out_dir / "assets" / "extropic-footer.mp4")
    else:
        missing.append(footer)
    if missing:
        detail = ", ".join(str(path) for path in missing)
        if _running_on_readthedocs():
            raise RuntimeError(
                f"licensed docs assets missing on Read the Docs: {detail}"
            )
        print(f"licensed assets unavailable; skipping local self-host copy: {detail}")


def _prepare_output(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text(_RENDERED_GITIGNORE, encoding="utf-8")


def _replace_output(staged_dir):
    backup_dir = None
    if OUT_DIR.exists():
        backup_dir = Path(tempfile.mkdtemp(dir=ROOT, prefix=".rendered-backup-"))
        backup_dir.rmdir()
        OUT_DIR.rename(backup_dir)
    try:
        staged_dir.rename(OUT_DIR)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not OUT_DIR.exists():
            backup_dir.rename(OUT_DIR)
        raise
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir)


def _smoke_check(out_dir, site, api_categories):
    required = [
        "index.html",
        "getting-started.html",
        "examples.html",
        "llms.txt",
        site.reading_order[0].href,
        "assets/extropic_wordmark.png",
        "assets/first_circuit.png",
    ]
    required.extend(LEGACY_REDIRECTS)  # every old api/<name>/index.html must be present
    required.extend(f"{category.slug}.html" for category in api_categories)
    missing = [path for path in required if not (out_dir / path).exists()]
    errors = [f"missing required output {path}" for path in missing]

    local_ipynb_link = re.compile(r'href="(?![a-z][a-z0-9+.-]*:|//)[^"]+\.ipynb', re.I)
    references_figures = False
    for html_path in out_dir.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        rel = html_path.relative_to(out_dir)
        if "<title>Notebook</title>" in text:
            errors.append(f"{rel}: leftover nbconvert title")
        if local_ipynb_link.search(text):
            errors.append(f"{rel}: local .ipynb link was not rewritten")
        if "vercel.app" in text or "__ASSET_CDN__" in text:
            errors.append(f"{rel}: external CDN / unresolved asset placeholder")
        if "assets/notebooks/" in text:
            references_figures = True
        # in-page anchors (the API symbol index, TOC links) must resolve
        ids = set(re.findall(r'\bid="([^"]+)"', text))
        for anchor in re.findall(r'href="#([^"]+)"', text):
            if anchor not in ids:
                errors.append(f"{rel}: dangling in-page anchor #{anchor}")
    # css/js can smuggle a CDN reference past the html-only sweep
    for asset_path in (*out_dir.rglob("*.css"), *out_dir.rglob("*.js")):
        text = asset_path.read_text(encoding="utf-8")
        if "vercel.app" in text or "__ASSET_CDN__" in text:
            rel = asset_path.relative_to(out_dir)
            errors.append(f"{rel}: external CDN / unresolved asset placeholder")
    # a build whose notebooks emit no figures is fine; only fail if pages
    # reference externalized figures but none were written. externalize_images
    # writes png/jpg/gif, so a jpeg/gif-only build still has figures on disk.
    fig_dir = out_dir / "assets" / "notebooks"
    figures_written = fig_dir.is_dir() and any(
        p.suffix in (".png", ".jpg", ".gif") for p in fig_dir.iterdir()
    )
    if references_figures and not figures_written:
        errors.append(
            "notebook pages reference externalized figures, but none were written"
        )
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"docs smoke check failed:\n{joined}")


def main():
    paths = notebook_paths()
    _fail_if_errors(validate_notebook_catalog(paths), "notebook catalog validation")
    validate_api_reference()

    exporter = HTMLExporter(template_name="lab", embed_images=True)
    api_categories = api_nav()

    notebooks = []
    excerpt_errors = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        excerpt_errors.extend(
            f"{path.name}: {error}" for error in validate_source_excerpts(nb, REPO_ROOT)
        )
        number = path.stem.split("_", 1)[0]
        title = notebook_title(nb, path.stem)
        # catalog validation above guarantees a complete docs block per notebook
        docs = nb["metadata"]["docs"]
        entry = NotebookEntry(
            number,
            title,
            path.stem,
            docs["blurb"],
            docs["section"],
            docs["featured"],
        )
        notebooks.append((path, nb, entry))
    _fail_if_errors(excerpt_errors, "source-excerpt validation")
    site = make_site(entry for _path, _nb, entry in notebooks)

    with tempfile.TemporaryDirectory(
        dir=ROOT, prefix=".rendered-", ignore_cleanup_errors=True
    ) as tmp:
        staged_dir = Path(tmp)
        _prepare_output(staged_dir)
        copy_brand_media(staged_dir)

        for path, nb, entry in notebooks:
            tag_hidden_inputs(nb)
            body, _ = exporter.from_notebook_node(nb)
            body = replace_once(
                body,
                "<title>Notebook</title>",
                f"<title>{html_lib.escape(entry.title)} &middot; Torx</title>",
                "notebook title marker",
            )
            body = normalize_cell_ids(body)
            body = externalize_images(body, path.stem, out_dir=staged_dir)
            body = rewrite_nb_links(body)
            body = linkify_api(body)
            body = linkify_helper_paths(body)
            body = inject_chrome(
                body, site, api_categories, active_stem=path.stem, title=entry.title
            )
            out = staged_dir / f"{path.stem}.html"
            out.write_text(body, encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)}  ({len(body) // 1024} KB)")

        write_doc_page(
            "getting-started",
            "Getting started",
            getting_started_inner(site),
            site,
            api_categories,
            "getting-started",
            out_dir=staged_dir,
        )
        write_doc_page(
            "examples",
            "Examples",
            examples_inner(site),
            site,
            api_categories,
            "examples",
            out_dir=staged_dir,
        )
        for category in api_categories:
            write_doc_page(
                category.slug,
                category.label,
                api_inner(
                    category.label, category.slug, category.blurb, category.symbols
                ),
                site,
                api_categories,
                None,
                mathjax=True,
                active_api=category.slug,
                out_dir=staged_dir,
            )
        write_index(site, out_dir=staged_dir)
        write_llms_txt(site, api_categories, out_dir=staged_dir)
        write_legacy_redirects(out_dir=staged_dir)
        pruned = prune_unreferenced_pngs(out_dir=staged_dir)
        if pruned:
            print(f"pruned {len(pruned)} unreferenced PNG assets")
        _smoke_check(staged_dir, site, api_categories)
        _replace_output(staged_dir)
    print(
        f"\n{len(site.reading_order)} notebooks + getting-started + {len(api_categories)} API pages + index "
        f"+ llms.txt rendered to {OUT_DIR.relative_to(ROOT)}/"
    )


if __name__ == "__main__":
    main()
