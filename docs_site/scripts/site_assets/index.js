
<script>
(function () {
  document.querySelectorAll('.tx-codecard .tx-copy').forEach(function (btn) {
    window.txWireCopy(btn, function () {
      var card = btn.closest('.tx-codecard');
      var pre = card ? card.querySelector('pre') : null;
      return pre ? pre.innerText : '';
    });
  });

  // subtle scroll-in for sections
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    document.querySelectorAll('.tx-reveal').forEach(function (el) { io.observe(el); });
  } else {
    document.querySelectorAll('.tx-reveal').forEach(function (el) { el.classList.add('in'); });
  }

  var c = document.getElementById('tx-led');
  if (!c) return;
  var ctx = c.getContext('2d');
  var COLS = 32, ROWS = 21;
  function resize() {
    var r = c.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
  }
  resize();
  var STOPS = [[0, 16, 9, 7], [0.30, 110, 40, 4], [0.55, 255, 132, 0], [0.78, 245, 214, 76], [1, 255, 241, 200]];
  function ramp(v) {
    if (v < 0) v = 0; if (v > 1) v = 1;
    for (var i = 1; i < STOPS.length; i++) {
      if (v <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i], f = (v - a[0]) / (b[0] - a[0]);
        return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f];
      }
    }
    return [255, 241, 200];
  }
  var motion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  var reduce = motion ? motion.matches : false;
  var onscreen = true, raf = 0;
  // Ising-inspired checkerboard wave across the dot grid.
  var spin = new Int8Array(COLS * ROWS), flash = new Float32Array(COLS * ROWS);
  for (var s = 0; s < spin.length; s++) spin[s] = Math.random() < 0.5 ? 1 : -1;
  var sweepX = -1, active = 0, BETA = 0.42;
  function nbsum(i, j) {
    return spin[j * COLS + (i + COLS - 1) % COLS] + spin[j * COLS + (i + 1) % COLS]
      + spin[((j + ROWS - 1) % ROWS) * COLS + i] + spin[((j + 1) % ROWS) * COLS + i];
  }
  function draw(animate) {
    var W = c.width, H = c.height, cw = W / COLS, ch = H / ROWS, R = Math.min(cw, ch) * 0.30;
    if (animate) {
      if (sweepX === -1) sweepX = -cw;
      sweepX += W / 135;
      if (sweepX > W + cw) { sweepX = -cw; active ^= 1; }
    }
    ctx.clearRect(0, 0, W, H);
    for (var j = 0; j < ROWS; j++) {
      for (var i = 0; i < COLS; i++) {
        var idx = j * COLS + i, cxp = (i + 0.5) * cw;
        var isActive = (((i + j) & 1) === active);
        if (animate && isActive && Math.abs(cxp - sweepX) < cw * 0.9) {
          var p1 = 1 / (1 + Math.exp(-2 * BETA * nbsum(i, j)));
          spin[idx] = Math.random() < p1 ? 1 : -1;
          flash[idx] = 1;
        }
        flash[idx] *= 0.88;
        var crest = Math.exp(-Math.pow((cxp - sweepX) / (cw * 1.8), 2));
        var lit = (spin[idx] + 1) * 0.5;
        var v = Math.min(1, lit * 0.4 + flash[idx] * 0.5 + crest * 0.5);
        var vv = Math.pow(v, 1.25);
        var col = ramp(vv);
        ctx.beginPath();
        if (vv > 0.55) { ctx.shadowColor = 'rgba(255,160,50,' + (vv * 0.85) + ')'; ctx.shadowBlur = R * 1.7; }
        else { ctx.shadowBlur = 0; }
        ctx.fillStyle = 'rgb(' + (col[0] | 0) + ',' + (col[1] | 0) + ',' + (col[2] | 0) + ')';
        ctx.arc(cxp, (j + 0.5) * ch, R * (0.4 + 0.6 * vv), 0, 6.2832);
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;
  }
  function canAnimate() {
    return !reduce && !document.hidden && onscreen;
  }
  function frame() {
    raf = 0;
    draw(true);
    if (canAnimate()) raf = window.requestAnimationFrame(frame);
  }
  function stop() {
    if (raf) window.cancelAnimationFrame(raf);
    raf = 0;
  }
  function refresh() {
    if (canAnimate()) {
      if (!raf) raf = window.requestAnimationFrame(frame);
    } else {
      stop();
      draw(false);
    }
  }
  window.addEventListener('resize', function () { resize(); draw(false); refresh(); });
  document.addEventListener('visibilitychange', refresh);
  if (motion) {
    var onMotion = function (e) { reduce = e.matches; refresh(); };
    if (motion.addEventListener) motion.addEventListener('change', onMotion);
    else if (motion.addListener) motion.addListener(onMotion);
  }
  if ('IntersectionObserver' in window) {
    var ledObserver = new IntersectionObserver(function (entries) {
      onscreen = entries[0] ? entries[0].isIntersecting : true;
      refresh();
    });
    ledObserver.observe(c);
  }
  draw(false);
  refresh();
})();
</script>
