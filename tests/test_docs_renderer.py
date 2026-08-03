# pyright: reportMissingImports=false
"""Focused contracts for the docs renderer."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "docs_site" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from render.api_docs import linkify_api
from render.chrome import build_notebook_actions, inject_chrome
from render.manifest import ApiCategory, NotebookEntry, Section, Site
from render.notebooks import (
    apply_figure_alts,
    linkify_helper_paths,
    validate_rendered_image_alts,
)
from render.pages import examples_inner, write_doc_page, write_index, write_llms_txt


class TestDocsRenderer(unittest.TestCase):
    def setUp(self):
        self.entries = (
            NotebookEntry(
                "01",
                "First notebook",
                "01_first_notebook",
                "First example.",
                "foundations",
                True,
            ),
            NotebookEntry(
                "02",
                "A complete notebook title that must remain available",
                "02_second_notebook",
                "Second example.",
                "foundations",
                False,
            ),
        )
        section = Section(
            "foundations", "Foundations", "Foundation examples.", self.entries
        )
        self.site = Site(self.entries, (section,), (self.entries[0],))

    def test_generated_pages_use_requested_brand_and_punctuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_doc_page(
                "examples",
                "Examples",
                examples_inner(self.site),
                self.site,
                (),
                "examples",
                out_dir=out_dir,
            )
            write_index(self.site, out_dir=out_dir)
            examples = (out_dir / "examples.html").read_text()
            landing = (out_dir / "index.html").read_text()

        self.assertIn("<title>Examples - Torx</title>", examples)
        self.assertIn(
            '<span class="tx-navcard-title">01 First notebook</span>',
            examples,
        )
        self.assertIn('href="https://extropic.ai/">EXTROPIC</a>', landing)
        self.assertIn("Chakra Petch", landing)
        self.assertNotIn("·", examples + landing)
        self.assertNotIn("&middot;", examples + landing)

    def test_llms_index_links_concrete_entry_points(self):
        category = ApiCategory("Core", "api-core", "Core primitives.", (), ("DFG",))
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_llms_txt(self.site, (category,), out_dir=out_dir)
            llms = (out_dir / "llms.txt").read_text()

        for target in (
            "https://docs.torx.ai/en/latest/index.html",
            "https://docs.torx.ai/en/latest/examples.html",
            "https://docs.torx.ai/en/latest/api-core.html",
            "https://github.com/extropic-ai/torx",
        ):
            self.assertIn(target, llms)
        self.assertNotIn("/api.html", llms)
        self.assertNotIn("Reading conventions", llms)

    def test_source_links_survive_nbconvert_markup(self):
        rel = "examples/helpers/_plots_sampling.py"
        expected = (
            '<a href="https://github.com/extropic-ai/torx/blob/main/'
            f'{rel}"><code>{rel}</code></a>'
        )
        for rendered in (
            f"<p>See <code>{rel}</code>.</p>",
            f"<p>See <code></code>{rel}.</p>",
        ):
            self.assertIn(expected, linkify_helper_paths(rendered))
        self.assertIn(
            f"{rel}c",
            linkify_helper_paths(f"<code></code>{rel}c"),
        )
        self.assertIn(
            "<code>Affine<wbr>Gaussian<wbr>Gate</code>",
            linkify_api("<code>AffineGaussianGate</code>"),
        )

    def test_notebook_chrome_links_source_despite_later_raw_marker(self):
        entry = self.entries[0]
        body = (
            "<html><head></head><body><main>"
            '<script>const marker = "</aside><main>";</script>'
            "</main></body></html>"
        )
        rendered = inject_chrome(
            body,
            self.site,
            (),
            active_stem=entry.stem,
            title=entry.title,
        )
        actions = build_notebook_actions(entry)

        self.assertIn(actions, rendered)
        self.assertIn(
            "github.com/extropic-ai/torx/blob/main/examples/01_first_notebook.ipynb",
            actions,
        )
        self.assertNotIn("Download notebook", actions)
        self.assertIn('content="First notebook - Torx"', rendered)

    def test_every_rendered_image_requires_an_authored_alt(self):
        data_uri = "data:image/png;base64,aGVsbG8="
        body = (
            '<div class="jp-Cell jp-CodeCell jp-Notebook-cell">'
            f"<img alt = 'figure' src=\"{data_uri}\"/>"
            "</div>"
        )
        notebook = SimpleNamespace(
            cells=[
                {"cell_type": "raw", "metadata": {}},
                {
                    "cell_type": "code",
                    "metadata": {"docs": {"alt": "Energy and cut value by step."}},
                },
            ]
        )

        rendered = apply_figure_alts(body, notebook)
        self.assertIn('alt="Energy and cut value by step."', rendered)
        externalized = rendered.replace(data_uri, "assets/notebooks/example.png")
        self.assertEqual(validate_rendered_image_alts(externalized), [])

        notebook.cells[1]["metadata"] = {}
        with self.assertRaisesRegex(RuntimeError, "metadata.docs.alt"):
            apply_figure_alts(body, notebook)

        for alt in (None, "", "No description has been provided for this image"):
            attribute = "" if alt is None else f" alt='{alt}'"
            invalid = f"<img src='assets/notebooks/example.png'{attribute}>"
            self.assertEqual(len(validate_rendered_image_alts(invalid)), 1)


if __name__ == "__main__":
    unittest.main()
