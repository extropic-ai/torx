
<script>
  (function () {
    var CHECK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    window.txWireCopy = function (btn, getText) {
      var original = btn.innerHTML;
      var restoreTimer = 0;
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        var text = getText();
        var done = function () {
          if (restoreTimer) window.clearTimeout(restoreTimer);
          btn.classList.add("tx-copied"); btn.innerHTML = CHECK;
          restoreTimer = window.setTimeout(function () { btn.classList.remove("tx-copied"); btn.innerHTML = original; }, 1300);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {});
        }
      });
    };
  })();
</script>
