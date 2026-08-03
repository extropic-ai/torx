"""API reference generation from the installed torx package.

The reference introspects the live ``torx`` package on every build, so it
tracks the current public API automatically. Each documented class renders its
members (methods, properties, and attributes with their signatures and
docstrings) in depth: ``inherited_members`` (members reachable through the MRO
are included), ``members_order: source`` (members ordered by source line), and
``show_if_no_docstring`` (members render even without a docstring). Category
membership is derived from each symbol's ``__module__`` (see
``_category_assignment``), so adding a gate or factor to torx surfaces it on the
right page automatically. Any public symbol whose module matches no curated
category is collected into an auto-generated "Additional API" page, and a
curated abstract base/member mismatch is fatal during validation, so API drift
fails before rendered output is replaced.
"""

import functools
import html as html_lib
import inspect
import math
import re
import sys
import types
import typing

import equinox

import torx
import torx.psc

from .manifest import API_CATEGORIES, API_PUBLIC_EXCLUSIONS, ApiCategory
from .text import span_already_linked

# distinguishes "no such attribute" from "exported as None"; bare `getattr(..., None) is None` conflates the two
_MISSING = object()

# modules whose public surface we document: `torx` is the factor-graph core, `torx.psc` the circuit stack
# we resolve a symbol from the first module that defines it, so a category list need not track re-exports
_API_MODULES = (torx, torx.psc)


def resolve_api_symbol(name):
    """Return the live object for ``name`` from the documented modules.

    Resolution walks ``_API_MODULES`` in order and returns the first match, so a
    name exported by both modules takes its top-level ``torx`` definition.
    Returns the ``_MISSING`` sentinel when no module exports the name.
    """
    for module in _API_MODULES:
        obj = getattr(module, name, _MISSING)
        if obj is not _MISSING:
            return obj
    return _MISSING


def _module_public_symbols(module):
    """Public names of one module, with re-exported submodules filtered out."""
    explicit = getattr(module, "__all__", None)
    if explicit is not None:
        names = set(explicit)
    else:
        names = {name for name in dir(module) if not name.startswith("_")}
    return {
        name
        for name in names
        if not isinstance(getattr(module, name, None), types.ModuleType)
    }


def _symbol_module(name):
    """The ``__module__`` of the live object for ``name`` (``""`` if absent)."""
    obj = resolve_api_symbol(name)
    return getattr(obj, "__module__", "") or ""


def _module_matches(module_name, prefixes):
    """True if ``module_name`` equals or is a submodule of any listed prefix.

    ``"torx.psc.gates._binary"`` matches prefix ``"torx.psc.gates"``;
    ``"torx.psc._circuit"`` matches the exact prefix ``"torx.psc._circuit"``.
    The ``prefix + "."`` guard keeps ``"torx.factor"`` from swallowing a
    hypothetical sibling like ``"torx.factorial"``.
    """
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in prefixes
    )


# sort key for top-level symbols: group by defining module, then source line, so
# a page's order tracks the package source rather than a hand-curated list
def _symbol_sort_key(name):
    obj = resolve_api_symbol(name)
    try:
        lineno = inspect.getsourcelines(typing.cast(typing.Any, obj))[1]
    except (OSError, TypeError):
        lineno = _UNLOCATABLE_LINENO
    return (getattr(obj, "__module__", "") or "", lineno, name)


@functools.cache
def _category_assignment():
    """Partition the public surface into ``{slug: [names]}`` plus leftovers.

    Each public symbol (minus ``API_PUBLIC_EXCLUSIONS``) is placed in the first
    category whose ``module_prefixes`` its ``__module__`` matches; categories
    are tried in manifest order. Symbols matching no category are returned
    separately and feed the auto "Additional API" page. Lists are source-ordered.
    """
    by_slug = {cat.slug: [] for cat in API_CATEGORIES}
    leftover = []
    for name in torx_public_symbols() - API_PUBLIC_EXCLUSIONS:
        module_name = _symbol_module(name)
        for cat in API_CATEGORIES:
            if _module_matches(module_name, cat.module_prefixes):
                by_slug[cat.slug].append(name)
                break
        else:
            leftover.append(name)
    for names in by_slug.values():
        names.sort(key=_symbol_sort_key)
    return by_slug, sorted(leftover, key=_symbol_sort_key)


def _is_abstract_symbol(name):
    """An abstract base by naming convention or by ``abc`` metaclass."""
    obj = resolve_api_symbol(name)
    return name.startswith("Abstract") or (
        inspect.isclass(obj) and inspect.isabstract(obj)
    )


def page_abstract_bases(slug):
    """Detected abstract bases for a page that renders an abstract subsection.

    Membership is module-derived and abstractness-detected; the per-base member
    list is curated by ``API_ABSTRACT_BASES`` (default: no members). Pages
    without an ``API_ABSTRACT_BASES`` entry get no subsection, so their abstract
    bases render inline in the main list, exactly as before.
    """
    members = dict(API_ABSTRACT_BASES.get(slug, ()))
    by_slug, _ = _category_assignment()
    return [
        (name, members.get(name, ()))
        for name in by_slug.get(slug, [])
        if _is_abstract_symbol(name)
    ]


def page_symbols(slug):
    """Main-list symbols for a page: module-matched, abstract bases removed when
    the page shows a dedicated abstract subsection."""
    by_slug, _ = _category_assignment()
    names = by_slug.get(slug, [])
    if slug in API_ABSTRACT_BASES:
        return [name for name in names if not _is_abstract_symbol(name)]
    return names


@functools.cache
def torx_public_symbols():
    """Public symbols across the documented modules, submodules filtered out.

    Cached so each module is scanned once per build. The public surfaces of
    top-level ``torx`` and ``torx.psc`` are unioned. Module-valued attributes
    (e.g. a re-exported ``import importlib`` or the ``psc``/``gates`` submodules)
    are dropped, so only real API symbols can ever reach the Additional API page
    and the manifest needs no per-module exclusion for them.
    """
    names = set()
    for module in _API_MODULES:
        names |= _module_public_symbols(module)
    return names


def additional_api_symbols():
    """Public torx symbols whose module matches no curated category.

    These are auto-included in the "Additional API" page so new torx API
    appears on the next build without a manifest edit. Returned in
    ``_symbol_sort_key`` order (the one source of truth set by
    ``_category_assignment``), so the warning and the page list agree; callers
    do not re-sort.
    """
    _by_slug, leftover = _category_assignment()
    return leftover


def validate_api_reference():
    """Validate curated API docs before rendered output is replaced.

    A public torx symbol whose module matches no curated category is reported
    and auto-included in the Additional API page. Curated aliases, abstract
    bases, and abstract members are fatal because they point at stale docs code.
    """
    undocumented = additional_api_symbols()
    if undocumented:
        print(
            "warning: public torx symbols are not in any curated category; "
            f"auto-including them in the Additional API page: {undocumented}",
            file=sys.stderr,
        )

    # Builds stage output, but stale curation should still fail before render work.
    errors = []
    for entries in API_ABSTRACT_BASES.values():
        for name, members in entries:
            obj = resolve_api_symbol(name)
            if obj is _MISSING:
                errors.append(f"API_ABSTRACT_BASES lists missing class {name}")
                continue
            for member in members:
                if member != "__init__" and member not in dir(obj):
                    errors.append(
                        f"API_ABSTRACT_BASES lists missing member {name}.{member}"
                    )
    try:
        api_symbol_url()
    except RuntimeError as exc:
        errors.append(str(exc))
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"API reference validation failed:\n{joined}")


ADDITIONAL_API_BLURB = (
    "Public torx symbols not yet sorted into a curated category. They are listed "
    "automatically so newly added API appears here on the next build."
)


def api_nav():
    """Curated API categories, plus an Additional API category on drift.

    The Additional API entry is appended only when a public symbol falls
    outside every curated category, so on a reconciled surface the navigation
    is exactly the curated categories.
    """
    pages = [
        cat._replace(symbols=tuple(page_symbols(cat.slug))) for cat in API_CATEGORIES
    ]
    extra = additional_api_symbols()
    if extra:
        pages.append(
            ApiCategory(
                "Additional API",
                "api-additional",
                ADDITIONAL_API_BLURB,
                module_prefixes=(),
                symbols=tuple(extra),
            )
        )
    return pages


# member curation for the "Abstract base classes" subsection; a slug here opts
# its page into the subsection; which abstract bases appear is derived from the
# package (see ``page_abstract_bases``), and these per-base member tuples pick
# which members each renders (a detected abstract base not listed shows none).
# pages absent from this map render their abstract bases inline in the main list
API_ABSTRACT_BASES = {
    "api-core": (),
    "api-circuits": (("AbstractPCircuit", ()),),
    "api-gates": (
        ("AbstractPGate", ()),
        ("AbstractGeneratorGate", ("dt", "get_generator", "get_matrix")),
        ("AbstractDiscreteGate", ("get_matrix",)),
        ("AbstractKBranchGate", ("num_branches", "probs", "branches")),
        ("AbstractSingleBinaryPGate", ()),
        ("AbstractMultiBinaryPGate", ()),
        ("AbstractSinglePditGate", ()),
        ("AbstractMultiPditGate", ()),
        ("AbstractHybridGate", ("sample",)),
        ("AbstractContinuousGate", ()),
        ("AbstractAffineGaussianGate", ("affine_parameters", "sample")),
        ("AbstractControlledContinuousGate", ()),
    ),
    "api-simulators": (
        ("AbstractSimulator", ("expval", "expval_all", "build_circuit")),
        ("AbstractCompiledPCircuit", ("from_pcircuit", "to_pcircuit")),
    ),
}


# prose writes PIsing; the exported symbol is PISING
API_SYMBOL_ALIASES = {"PIsing": "PISING"}

API_DOC_OVERRIDES = {
    "HybridSites": (
        "Site mapping for hybrid gates.\n\n"
        "**Keys:**\n\n"
        "- `discrete`: discrete site indices read as controls.\n"
        "- `continuous`: continuous site indices updated by the gate.\n\n"
        "Continuous-only gates also accept a list of continuous site indices; "
        "hybrid gate constructors accept a `(discrete, continuous)` pair."
    ),
}

API_SIGNATURE_OVERRIDES = {
    "HybridSites": "(discrete: list[int], continuous: list[int])",
}


@functools.cache
def api_symbol_url():
    """Map each documented symbol to its ``<page>.html#<symbol>`` anchor.

    Built lazily (and cached) so importing this module triggers no torx
    introspection via ``api_nav()``. Includes the auto-generated Additional API
    page, so a cross-reference or alias can never KeyError against a
    curated-only map.
    """
    url = {name: f"{cat.slug}.html#{name}" for cat in api_nav() for name in cat.symbols}
    # abstract bases render in their page's subsection (anchor id="{name}"), but
    # are kept out of `cat.symbols`; add them so xrefs and autorefs can link them
    for cat in api_nav():
        for name, _members in page_abstract_bases(cat.slug):
            url[name] = f"{cat.slug}.html#{name}"
    stale = set(API_SYMBOL_ALIASES.values()) - set(url)
    if stale:
        raise RuntimeError(
            f"API_SYMBOL_ALIASES targets are not documented API symbols: {sorted(stale)}"
        )
    return url


@functools.cache
def _api_xref_re():
    url = api_symbol_url()
    return re.compile(
        r"<code>("
        + "|".join(
            re.escape(n)
            for n in sorted(set(url) | set(API_SYMBOL_ALIASES), key=len, reverse=True)
        )
        + r")</code>"
    )


def linkify_api(html):
    """Link bare API-name code spans in notebook prose to the API reference.

    Only matches whole ``<code>Name</code>`` spans (markdown inline code), so
    code cell source and outputs, which use different markup, are left
    untouched. A code span already inside an anchor is skipped so links
    never nest.
    """
    url = api_symbol_url()

    def repl(m):
        if span_already_linked(html, m.start()):
            return m.group(0)
        name = typing.cast(str, m.group(1))
        target = API_SYMBOL_ALIASES.get(name, name)
        display = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "<wbr>", name)
        return f'<a class="api-xref" href="{url[target]}"><code>{display}</code></a>'

    return _api_xref_re().sub(repl, html)


# module-path noise stripped from rendered types; shared so signature and annotation paths never diverge
_TYPE_NOISE_PATTERNS = (
    r"jax\.jaxlib\._jax\.",
    r"jaxtyping\.",
    r"jaxlib\._jax\.",
    r"collections\.abc\.",
    r"torx\.[\w.]+\.",
)


def _strip_type_noise(s):
    for pat in _TYPE_NOISE_PATTERNS:
        s = re.sub(pat, "", s)
    return s


_SIG_WRAP_WIDTH = 72  # one-line sigs longer than this render one param per line


def _clean_signature(obj):
    """Render a callable's signature, dropping the bound receiver parameter.

    Long signatures wrap one parameter per line so the API card never needs a
    horizontal scrollbar. Parameters come from ``inspect`` (each keeps its own
    annotation intact, commas and all); only the return tail is sliced off by
    bracket-matching the depth-0 close paren, never by splitting on commas.
    """
    try:
        sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
    except (ValueError, TypeError):
        return "(...)"
    params = list(sig.parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    sig = sig.replace(parameters=params)
    oneline = re.sub(r" -> None$", "", _strip_type_noise(str(sig)))
    if len(oneline) <= _SIG_WRAP_WIDTH:
        return oneline
    depth = 0
    close = len(oneline) - 1
    for i, ch in enumerate(oneline):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close = i
                break
    tail = oneline[close + 1 :]
    # rebuild params one per line, reinserting the positional-only `/` and
    # keyword-only `*` markers that are NOT Parameter objects (so per-param
    # str() drops them) — mirrors inspect.Signature.__str__ separator logic.
    rendered = []
    emit_kw_star = True  # bare `*` before the first KEYWORD_ONLY, unless *args seen
    prev_kind = None
    for p in params:
        if (
            prev_kind == inspect.Parameter.POSITIONAL_ONLY
            and p.kind != inspect.Parameter.POSITIONAL_ONLY
        ):
            rendered.append("/")
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            emit_kw_star = False
        elif p.kind == inspect.Parameter.KEYWORD_ONLY and emit_kw_star:
            rendered.append("*")
            emit_kw_star = False
        rendered.append(_strip_type_noise(str(p)))
        prev_kind = p.kind
    if prev_kind == inspect.Parameter.POSITIONAL_ONLY:
        rendered.append("/")
    body = ",\n    ".join(rendered)
    return f"(\n    {body},\n){tail}"


def _annotation_text(annotation):
    """Render a field annotation to a compact string, preserving subscripts.

    ``str(annotation)`` keeps subscripts (``dict[str, Array]``,
    ``tuple[int, ...]``) that ``__name__`` would have dropped; the module-noise
    denylist then strips the long dotted paths.
    """
    s = annotation if isinstance(annotation, str) else str(annotation)
    s = re.sub(r"<class '([^']+)'>", r"\1", s)
    s = _strip_type_noise(s.replace("typing.", ""))
    return s.strip()


def _render_inline(s):
    # input is trusted torx docstring text; `quote=False` keeps prose punctuation unescaped
    # escape before markup so the emitted <code>/<strong> tags survive
    s = html_lib.escape(s, quote=False)
    s = re.sub(r"\[`([^`]+)`\]\[\]", _render_autoref, s)
    s = re.sub(r"``([^`]+)``", r"<code>\1</code>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+?)\*\*", r"<strong>\1</strong>", s)
    return s


def _render_autoref(match):
    ref = typing.cast(str, match.group(1))
    urls = api_symbol_url()
    parts = ref.split(".")
    # resolve through the symbol aliases, exactly like linkify_api
    cls_target = API_SYMBOL_ALIASES.get(parts[-2], parts[-2]) if len(parts) >= 2 else ""
    name_target = API_SYMBOL_ALIASES.get(parts[-1], parts[-1])
    if cls_target in urls:
        cls_name, member = parts[-2], parts[-1]
        page = urls[cls_target].split("#", 1)[0]
        display = f"{cls_name}.{member}"
        # the constructor renders inside the class section without an
        # `__init__` anchor (see `_render_member`/`_render_class`), so link the
        # class section itself rather than a `#Class.__init__` that never exists
        anchor = cls_target if member == "__init__" else f"{cls_target}.{member}"
        return (
            f'<a class="api-xref" href="{page}#{anchor}">'
            f"<code>{html_lib.escape(display, quote=False)}</code></a>"
        )
    if name_target in urls:
        name = parts[-1]
        return (
            f'<a class="api-xref" href="{urls[name_target]}">'
            f"<code>{html_lib.escape(name, quote=False)}</code></a>"
        )
    display = _strip_type_noise(ref)
    return f"<code>{html_lib.escape(display, quote=False)}</code>"


# sentinel wrapping each stashed LaTeX span (`\x00M{n}\x00`); NUL can't occur in a docstring
# so the block parser treats a math span as opaque and the final pass restores LaTeX by index
_MATH_SENTINEL_RE = re.compile(r"^\x00M\d+\x00$")

# markdown pipe table: a row has at least one `|`, the separator row is all dashes/pipes/colons
_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")


def _table_cells(row):
    """Split a ``|a|b|`` table row into its trimmed cell texts."""
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _render_markdown_blocks(lines):
    """Parse light markdown into HTML: paragraphs, ``- `` lists, ``???``
    admonitions, and display-math lines. Math sentinels are left in place for
    the caller to restore once."""
    out, para = [], []
    in_list = False

    def flush():
        if para:
            out.append("<p>" + _render_inline(" ".join(para)) + "</p>")
            para.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("```"):
            flush()
            close_list()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence (or end of input)
            body = html_lib.escape("\n".join(code))
            out.append(f'<pre class="api-code"><code>{body}</code></pre>')
            continue
        if (
            _TABLE_ROW_RE.match(ln)
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            flush()
            close_list()
            header = _table_cells(ln)
            i += 2  # header + separator
            rows = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                rows.append(_table_cells(lines[i].strip()))
                i += 1
            head = "".join(f"<th>{_render_inline(c)}</th>" for c in header)
            cells = "".join(
                "<tr>" + "".join(f"<td>{_render_inline(c)}</td>" for c in row) + "</tr>"
                for row in rows
            )
            out.append(
                f'<table class="api-table"><thead><tr>{head}</tr></thead>'
                f"<tbody>{cells}</tbody></table>"
            )
            continue
        admon = re.match(
            r"(?P<mark>\?{3}|!{3})\+?\s*(?P<kind>\w+)"
            r'(?:\s+"?(?P<title>[^"]*)"?\s*)?$',
            ln,
        )
        if admon:
            flush()
            close_list()
            title = admon.group("title") or admon.group("kind").title()
            i += 1
            body = []
            while i < len(lines) and (
                lines[i].strip() == "" or lines[i].startswith("    ")
            ):
                body.append(lines[i][4:] if lines[i].startswith("    ") else "")
                i += 1
            inner = _render_markdown_blocks(body)
            out.append(
                f'<div class="note"><span class="note-t">{_render_inline(title)}</span>{inner}</div>'
            )
            continue
        if _MATH_SENTINEL_RE.match(ln):
            flush()
            close_list()
            out.append(f'<div class="api-display">{ln}</div>')
            i += 1
            continue
        if ln.startswith("- "):
            flush()
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = [ln[2:]]
            i += 1
            while (
                i < len(lines)
                and lines[i].startswith("    ")
                and not lines[i].strip().startswith("- ")
            ):
                item.append(lines[i].strip())
                i += 1
            out.append("<li>" + _render_inline(" ".join(item)) + "</li>")
            continue
        if ln == "":
            flush()
            close_list()
            i += 1
            continue
        para.append(ln)
        i += 1
    flush()
    close_list()
    return "\n".join(out)


def render_docstring(doc):
    """Render a torx docstring (light markdown + LaTeX) to themed HTML for MathJax."""
    math_spans = []

    def stash(m):
        math_spans.append(m.group(0))
        return f"\x00M{len(math_spans) - 1}\x00"

    doc = re.sub(r"\$\$.*?\$\$", stash, doc, flags=re.S)
    doc = re.sub(r"\$[^$\n]+?\$", stash, doc)
    res = _render_markdown_blocks(doc.split("\n"))
    for k, mx in enumerate(math_spans):
        res = res.replace(f"\x00M{k}\x00", mx)
    return res


def _member_target(static):
    """Unwrap a descriptor to the callable that carries the signature/docstring."""
    if isinstance(static, property):
        return static.fget
    if isinstance(static, functools.cached_property):
        return static.func
    if isinstance(static, (staticmethod, classmethod)):
        return static.__func__
    return static


# sort key for members with no locatable source line (builtins, C-level); sort them after locatable ones
_UNLOCATABLE_LINENO = math.inf


def _static_member(cls, name):
    """The unbound member descriptor, falling back to a dynamic attribute.

    ``getattr_static`` avoids triggering descriptors, but misses members a
    metaclass synthesizes; the ``getattr`` fallback resolves those.
    """
    try:
        return inspect.getattr_static(cls, name)
    except AttributeError:
        return getattr(cls, name, None)


def _member_lineno(cls, name):
    static = _static_member(cls, name)
    target = _member_target(static)
    try:
        return inspect.getsourcelines(typing.cast(typing.Any, target))[1]
    except (OSError, TypeError):
        return _UNLOCATABLE_LINENO


def _method_names(cls):
    """Public methods and properties on cls, including inherited, source-ordered."""
    names = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            static = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        is_member = (
            isinstance(
                static, (property, functools.cached_property, staticmethod, classmethod)
            )
            or inspect.isfunction(static)
            or inspect.ismethod(static)
        )
        if is_member:
            names.append(name)
    return sorted(names, key=lambda n: _member_lineno(cls, n))


def _is_abstract_annotation(ftype):
    """True if the annotation marks an abstract or class-level field that is not
    a concrete instance attribute (``equinox.AbstractVar`` / ``typing.ClassVar``)."""
    if ftype is equinox.AbstractVar or ftype is typing.ClassVar:
        return True
    if typing.get_origin(ftype) in (equinox.AbstractVar, typing.ClassVar):
        return True
    if isinstance(ftype, str) and (
        ftype.startswith("AbstractVar") or ftype.startswith("ClassVar")
    ):
        return True
    return False


def _public_fields(cls):
    """Public concrete annotated fields across the MRO (equinox/dataclass attrs).

    Abstract interface annotations (``AbstractVar``) and class-level
    ``ClassVar`` annotations are skipped, as are names a subclass overrides with
    a property/descriptor, so an abstract ``dims: AbstractVar`` never leaks in
    and shadows the concrete ``dims`` property.
    """
    seen = set()
    fields = []
    for klass in inspect.getmro(cls):
        if klass is object:
            continue
        for fname, ftype in getattr(klass, "__annotations__", {}).items():
            if fname.startswith("_") or fname in seen:
                continue
            seen.add(fname)
            if _is_abstract_annotation(ftype):
                continue
            try:
                static = inspect.getattr_static(cls, fname)
            except AttributeError:
                static = None
            if isinstance(
                static, (property, functools.cached_property)
            ) or inspect.isdatadescriptor(static):
                continue
            fields.append((fname, ftype))
    return fields


def _render_field(cls_name, fname, ftype):
    ann = html_lib.escape(_annotation_text(ftype))
    name = html_lib.escape(fname)
    return (
        f'<div class="api-member" id="{cls_name}.{fname}">'
        f'<div class="api-head"><code class="api-name">{name}</code>'
        f'<span class="api-kind">attribute</span></div>'
        f'<pre class="api-sig"><code>{name}: {ann}</code></pre>'
        "</div>"
    )


def _render_member(cls, name):
    static = _static_member(cls, name)
    target = _member_target(static)
    if isinstance(static, (property, functools.cached_property)):
        kind, sig = "property", ""
    elif isinstance(static, staticmethod):
        kind, sig = "staticmethod", _clean_signature(target)
    elif isinstance(static, classmethod):
        kind, sig = "classmethod", _clean_signature(target)
    else:
        kind, sig = "method", _clean_signature(target)
    doc = inspect.getdoc(target) or ""
    head = html_lib.escape(name + sig)
    body = render_docstring(doc) if doc else ""
    return (
        f'<div class="api-member" id="{cls.__name__}.{name}">'
        f'<div class="api-head"><code class="api-name">{html_lib.escape(name)}</code>'
        f'<span class="api-kind">{kind}</span></div>'
        f'<pre class="api-sig"><code>{head}</code></pre>'
        f'<div class="api-doc">{body}</div>'
        "</div>"
    )


def _render_constructor_doc(cls):
    if typing.is_typeddict(cls):
        return ""
    target = _member_target(_static_member(cls, "__init__"))
    doc = inspect.getdoc(target) or ""
    if not doc or doc == inspect.getdoc(object.__init__):
        return ""
    return (
        '<div class="api-constructor">'
        '<span class="api-kind">constructor</span>'
        f'<div class="api-doc">{render_docstring(doc)}</div>'
        "</div>"
    )


def _render_class(cls_name, obj, explicit_members=None):
    name = html_lib.escape(cls_name)
    sig = html_lib.escape(
        cls_name + API_SIGNATURE_OVERRIDES.get(cls_name, _clean_signature(obj))
    )
    doc = API_DOC_OVERRIDES.get(cls_name, inspect.getdoc(obj) or "")
    parts = [
        f'<section class="api-sym" id="{name}">'
        f'<div class="api-head"><code class="api-name">{name}</code>'
        f'<span class="api-kind">class</span></div>'
        f'<pre class="api-sig"><code>{sig}</code></pre>'
        f'<div class="api-doc">{render_docstring(doc)}</div>'
    ]
    constructor = _render_constructor_doc(obj)
    if constructor:
        parts.append(constructor)

    member_html = []
    if explicit_members is None:
        fields = _public_fields(obj)
        field_names = {fname for fname, _ in fields}
        for fname, ftype in fields:
            member_html.append(_render_field(cls_name, fname, ftype))
        for member in _method_names(obj):
            if member in field_names:
                continue
            member_html.append(_render_member(obj, member))
    else:
        for member in explicit_members:
            if member == "__init__":
                continue
            if member not in dir(obj):
                raise RuntimeError(
                    f"API_ABSTRACT_BASES lists {cls_name}.{member}, "
                    f"but {cls_name} has no such member"
                )
            member_html.append(_render_member(obj, member))

    if member_html:
        parts.append('<div class="api-members">' + "".join(member_html) + "</div>")
    parts.append("</section>")
    return "".join(parts)


def _render_function(name, obj):
    safe = html_lib.escape(name)
    sig = html_lib.escape(name + _clean_signature(obj))
    doc = inspect.getdoc(obj) or ""
    return (
        f'<section class="api-sym" id="{safe}">'
        f'<div class="api-head"><code class="api-name">{safe}</code>'
        f'<span class="api-kind">function</span></div>'
        f'<pre class="api-sig"><code>{sig}</code></pre>'
        f'<div class="api-doc">{render_docstring(doc)}</div>'
        "</section>"
    )


def api_inner(label, slug, blurb, symbols):
    parts = [f"<h1>{label}</h1>\n", f'<p class="lede">{blurb}</p>\n']

    abstract = page_abstract_bases(slug)
    # symbol index: wayfinding for long pages; skipped when every entry
    # already fits on one screen
    index_names = [name for name, _members in abstract] + list(symbols)
    if len(index_names) > 6:
        links = "".join(
            f'<a href="#{html_lib.escape(name, quote=True)}">'
            f"<code>{html_lib.escape(name, quote=False)}</code></a>"
            for name in index_names
        )
        parts.append(f'<nav class="api-index" aria-label="Symbols">{links}</nav>\n')

    if abstract:
        parts.append('<section class="api-group"><h2>Abstract base classes</h2>')
        for cls_name, members in abstract:
            obj = resolve_api_symbol(cls_name)
            # `validate_api_reference` (run in main before any render) is the one
            # place that reports a curated abstract base missing from torx; skip
            # it silently here rather than warning a second time
            if obj is _MISSING:
                continue
            # an empty curated tuple means "no curation": render the base's full
            # auto-detected members, like a concrete class, instead of nothing
            parts.append(
                _render_class(cls_name, obj, explicit_members=list(members) or None)
            )
        parts.append("</section>")

    # `symbols` come from live introspection of the package, so they always
    # resolve; separate heading so concrete entries don't read as abstract bases
    if abstract:
        parts.append('<section class="api-group"><h2>Concrete classes</h2>')
    for name in symbols:
        obj = resolve_api_symbol(name)
        if inspect.isclass(obj):
            parts.append(_render_class(name, obj))
        else:
            parts.append(_render_function(name, obj))
    if abstract:
        parts.append("</section>")
    return "".join(parts)
