# pyright: reportMissingImports=false
"""Focused contracts for the docs renderer."""

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "docs_site" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


from render.api_docs import linkify_api
from render.assets import THEME_CSS
from render.chrome import (
    build_notebook_actions,
    inject_chrome,
    prev_next_nav,
)
from render.manifest import ApiCategory, NotebookEntry, Section, Site
from render.notebooks import (
    apply_figure_alts,
    linkify_helper_paths,
    validate_rendered_image_alts,
)
from render.pages import (
    examples_inner,
    write_doc_page,
    write_index,
    write_llms_txt,
)


class TestDocsRenderer(unittest.TestCase):
    def setUp(self):
        self.entries = tuple(
            NotebookEntry(
                f"{number:02d}",
                title,
                f"{number:02d}_{title.lower().replace(' ', '_')}",
                f"About {title.lower()}.",
                "foundations",
                number == 1,
            )
            for number, title in (
                (1, "First notebook"),
                (2, "A complete notebook title that must remain available"),
                (3, "Third notebook"),
            )
        )
        section = Section(
            "foundations", "Foundations", "Foundation examples.", self.entries
        )
        self.site = Site(self.entries, (section,), (self.entries[0],))

    def test_page_metadata_and_example_cards_do_not_use_interpuncts(self):
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
            examples = (out_dir / "examples.html").read_text(encoding="utf-8")
            landing = (out_dir / "index.html").read_text(encoding="utf-8")

        self.assertIn("<title>Examples - Torx</title>", examples)
        self.assertIn('property="og:title" content="Examples - Torx"', examples)
        self.assertIn('<span class="tx-navcard-number">01</span>', examples)
        self.assertIn('<span class="tx-navcard-title">First notebook</span>', examples)
        self.assertIn("<title>Torx - Parametrised Stochastic Circuits</title>", landing)
        self.assertIn(
            '<a class="tx-foot-extropic" href="https://extropic.ai/">EXTROPIC</a>',
            landing,
        )
        self.assertNotIn("·", examples + landing)
        self.assertNotIn("&middot;", examples + landing)

    def test_llms_index_exposes_entry_points_and_execution_conventions(self):
        category = ApiCategory(
            "Core",
            "api-core",
            "Core primitives.",
            (),
            ("DFG",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_llms_txt(self.site, (category,), out_dir=out_dir)
            llms = (out_dir / "llms.txt").read_text(encoding="utf-8")

        self.assertIn("## Start here", llms)
        self.assertIn(
            "https://docs.torx.ai/en/latest/examples.html",
            llms,
        )
        self.assertIn(
            "https://docs.torx.ai/en/latest/api-core.html",
            llms,
        )
        self.assertNotIn("https://docs.torx.ai/en/latest/api.html", llms)
        self.assertIn("https://github.com/extropic-ai/torx", llms)
        self.assertIn("## Reading conventions", llms)
        self.assertIn("rendered from executed notebooks", llms)
        self.assertIn("Exact, sampled, and host-side results", llms)

    def test_notebook_actions_use_configured_repository_paths(self):
        entry = self.entries[0]
        source_path = f"examples/{entry.stem}.ipynb"
        actions = build_notebook_actions(entry)

        self.assertIn(
            f'href="https://github.com/extropic-ai/torx/blob/main/{source_path}"',
            actions,
        )
        self.assertIn(
            (
                'href="https://github.com/extropic-ai/torx/raw/main/'
                f'{source_path}" download'
            ),
            actions,
        )
        self.assertIn("View source", actions)
        self.assertIn("Download notebook", actions)

        rendered = inject_chrome(
            "<html><head></head><body><main></main></body></html>",
            self.site,
            (),
            active_stem=entry.stem,
            title=entry.title,
        )
        self.assertIn(actions, rendered)
        self.assertIn('property="og:title" content="First notebook - Torx"', rendered)

    def test_helper_paths_link_when_nbconvert_detaches_inline_code_tag(self):
        rel = "examples/helpers/_plots_sampling.py"
        expected = (
            '<a href="https://github.com/extropic-ai/torx/blob/main/'
            f'{rel}"><code>{rel}</code></a>'
        )
        for rendered in (
            f"<p>See <code>{rel}</code>.</p>",
            f"<p>See <code></code>{rel}.</p>",
        ):
            with self.subTest(rendered=rendered):
                linked = linkify_helper_paths(rendered)
                self.assertIn(expected, linked)
                self.assertNotIn(f"<code></code>{rel}", linked)

    def test_long_api_names_break_only_at_camel_case_boundaries(self):
        linked = linkify_api("<p><code>AffineGaussianGate</code></p>")
        self.assertIn(
            "<code>Affine<wbr>Gaussian<wbr>Gate</code>",
            linked,
        )

    def test_cell_docs_alt_propagates_and_placeholders_fail(self):
        data_uri = "data:image/png;base64,aGVsbG8="
        body = (
            '<div class="jp-Cell jp-CodeCell jp-Notebook-cell">'
            '<img alt="No description has been provided for this image" '
            f'src="{data_uri}"/>'
            "</div>"
        )
        notebook = SimpleNamespace(
            cells=[
                {
                    "cell_type": "code",
                    "metadata": {
                        "docs": {
                            "alt": (
                                "Energy and cut value by training step, "
                                "with the best step marked"
                            )
                        }
                    },
                }
            ]
        )

        rendered = apply_figure_alts(body, notebook)
        self.assertIn(
            'alt="Energy and cut value by training step, with the best step marked"',
            rendered,
        )
        externalized = rendered.replace(data_uri, "assets/notebooks/example.png")
        self.assertEqual(validate_rendered_image_alts(externalized), [])

        placeholder = externalized.replace(
            "Energy and cut value by training step, with the best step marked",
            "No description has been provided for this image",
        )
        self.assertEqual(len(validate_rendered_image_alts(placeholder)), 1)

    def test_previous_next_titles_wrap_and_keep_full_link_text(self):
        navigation = prev_next_nav(self.site, self.entries[0].stem)
        self.assertIn(self.entries[1].title, navigation)

        title_rule = re.search(r"\.tx-pn-title\s*\{([^}]+)\}", THEME_CSS)
        assert title_rule is not None
        declarations = title_rule.group(1)
        self.assertIn("white-space: normal", declarations)
        self.assertNotIn("text-overflow: ellipsis", declarations)
        self.assertNotIn("overflow: hidden", declarations)
        self.assertIn(".tx-pagenav { flex-direction: column; }", THEME_CSS)


if __name__ == "__main__":
    unittest.main()
