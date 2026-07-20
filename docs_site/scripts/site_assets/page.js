
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

    // hidden source cells fold behind a pill toggle; consecutive hidden cells share one pill
    var hiddenCells = Array.prototype.slice.call(
      document.querySelectorAll(".jp-CodeCell.celltag_hide-input")
    );
    var groups = [];
    hiddenCells.forEach(function (cell) {
      var prev = groups[groups.length - 1];
      if (prev && prev[prev.length - 1].nextElementSibling === cell) prev.push(cell);
      else groups.push([cell]);
    });
    groups.forEach(function (group, i) {
      var first = group[0];
      if (first.querySelector(".tx-toggle")) return;
      var ids = group.map(function (cell, j) {
        var wrapper = cell.querySelector(".jp-Cell-inputWrapper");
        if (wrapper && !wrapper.id) wrapper.id = "tx-hidden-code-" + i + "-" + j;
        return wrapper ? wrapper.id : null;
      }).filter(Boolean);
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tx-toggle";
      btn.title = "Toggle hidden code";
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-controls", ids.join(" "));
      var label = document.createElement("span");
      label.textContent = "show code";
      btn.innerHTML = '<span class="tx-chevron">' + CHEV + "</span>";
      btn.appendChild(label);
      btn.addEventListener("click", function () {
        var open = !first.classList.contains("tx-open");
        group.forEach(function (cell) { cell.classList.toggle("tx-open", open); });
        label.textContent = open ? "hide code" : "show code";
        btn.setAttribute("aria-expanded", open ? "true" : "false");
      });
      first.insertBefore(btn, first.firstChild);
    });

    // scroll the sidebar to the active entry (API pages land far down)
    var side = document.querySelector(".tx-sidebar");
    var act = side ? side.querySelector(".active") : null;
    if (side && act) {
      var delta = act.getBoundingClientRect().top - side.getBoundingClientRect().top;
      // scroll only when the entry sits below the fold (80px = one entry of slack), centering it
      if (delta > side.clientHeight - 80) side.scrollTop = delta - side.clientHeight / 2;
    }

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
