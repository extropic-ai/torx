
<script>
  document.addEventListener("DOMContentLoaded", function () {
    var COPY = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    var CHEV = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 6 15 12 9 18"/></svg>';

    // External links open in a new tab; internal site/anchor links stay in place.
    document.querySelectorAll("a[href]").forEach(function (a) {
      var h = a.getAttribute("href") || "";
      var url;
      try { url = new URL(h, location.href); } catch (e) { return; }
      if (url.origin !== location.origin && /^https?:$/.test(url.protocol)) {
        a.target = "_blank";
        var rel = new Set((a.getAttribute("rel") || "").split(/\s+/).filter(Boolean));
        rel.add("noopener");
        a.setAttribute("rel", Array.from(rel).join(" "));
      }
    });

    // Notebook code cells: header bar (language chip + copy).
    document.querySelectorAll(".jp-CodeCell .jp-InputArea-editor").forEach(function (editor) {
      if (editor.querySelector(".tx-code-head")) return;
      var head = document.createElement("div");
      head.className = "tx-code-head";
      head.innerHTML = '<span class="tx-lang"></span>' +
        '<button class="tx-copy" type="button" title="Copy code" aria-label="Copy code">' + COPY + '</button>';
      editor.insertBefore(head, editor.firstChild);
      var sourcePre = editor.querySelector(".highlight pre");
      if (!sourcePre) return;
      window.txWireCopy(head.querySelector(".tx-copy"), function () {
        return sourcePre.textContent;
      });
    });

    // Docs-page code cards.
    document.querySelectorAll(".tx-codecard .tx-copy").forEach(function (btn) {
      var card = btn.closest(".tx-codecard");
      var pre = card ? card.querySelector("pre") : null;
      if (!card || !pre) return;
      if (!btn.innerHTML.trim()) btn.innerHTML = COPY;
      window.txWireCopy(btn, function () { return pre.textContent; });
    });

    // Hidden source cells fold behind a pill toggle.
    document.querySelectorAll(".jp-CodeCell.celltag_hide-input").forEach(function (cell, i) {
      if (cell.querySelector(".tx-toggle")) return;
      var wrapper = cell.querySelector(".jp-Cell-inputWrapper");
      if (!wrapper) return;
      if (!wrapper.id) wrapper.id = "tx-hidden-code-" + i;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tx-toggle";
      btn.title = "Toggle setup";
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-controls", wrapper.id);
      btn.innerHTML = '<span class="tx-chevron">' + CHEV + '</span><span>setup</span>';
      btn.addEventListener("click", function () {
        var open = cell.classList.toggle("tx-open");
        btn.setAttribute("aria-expanded", open ? "true" : "false");
      });
      cell.insertBefore(btn, cell.firstChild);
    });

    // Mobile sidebar toggle.
    var burger = document.querySelector(".tx-burger");
    var sb = document.querySelector(".tx-sidebar");
    if (burger && sb) {
      if (!sb.id) sb.id = "tx-sidebar";
      burger.setAttribute("aria-controls", sb.id);
      burger.setAttribute("aria-expanded", "false");
      var bd = document.createElement("div");
      bd.className = "tx-backdrop";
      document.body.appendChild(bd);
      var HAM = burger.innerHTML;
      var XICON = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>';
      function txSetNav(open) {
        sb.classList.toggle("tx-open", open); bd.classList.toggle("tx-open", open);
        burger.innerHTML = open ? XICON : HAM;
        burger.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
        burger.setAttribute("aria-expanded", open ? "true" : "false");
      }
      burger.addEventListener("click", function () { txSetNav(!sb.classList.contains("tx-open")); });
      bd.addEventListener("click", function () { txSetNav(false); });
    }
  });
</script>
