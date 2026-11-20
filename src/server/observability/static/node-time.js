// Time-axis drag, Ctrl/⌘+wheel zoom, two-finger pinch, and W/S A/D navigation.
import { MAX_WINDOW, now, rangeOf, setCrosshair, ui, view } from "./state.js";

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const SURFACES = ".capacity-track, .node-axis, .activity-track, .activity-axis, #placement-detail .chart-plot svg, #placement-detail .chart-empty";

export function attachNodeTime(root) {
  let gesture = null, drag = null, pinch = null, settle, ignoreClickUntil = 0;
  let pointerAt = null;
  const getNow = () => now(ui.inventory.data);
  const getRange = () => rangeOf(ui.inventory.data, view);
  const fraction = (x, box) => clamp((x - box.left) / (box.width || 1), 0, 1);

  function begin(kind) {
    if (gesture) return;
    gesture = { kind, duration: view.duration, end: view.end };
    ui.navigating = true;
    root.classList.add("node-navigating");
  }

  function select(duration, end) {
    duration = clamp(Math.round(duration), 60, MAX_WINDOW);
    view.duration = duration;
    view.end = clamp(end, getNow() - MAX_WINDOW + duration, getNow());
    ui.render();
  }

  function finish(cancel = false) {
    clearTimeout(settle);
    if (drag?.pointerId !== undefined && root.hasPointerCapture(drag.pointerId)) root.releasePointerCapture(drag.pointerId);
    drag = pinch = null;
    if (!gesture) return;
    if (cancel) {
      view.duration = gesture.duration;
      view.end = gesture.end;
    }
    if (gesture.kind === "drag") ignoreClickUntil = performance.now() + 400;
    gesture = null;
    ui.navigating = false;
    root.classList.remove("node-navigating");
    ui.render();
  }

  function moveDrag(x, y, touch = false) {
    if (!drag) return false;
    const dx = x - drag.x, dy = y - drag.y;
    if (!gesture) {
      if (touch && Math.abs(dy) > 6 && Math.abs(dy) > Math.abs(dx)) { drag = null; return false; }
      if (Math.abs(dx) < 6) return false;
      begin("drag");
      if (drag.pointerId !== undefined) root.setPointerCapture(drag.pointerId);
    }
    select(drag.range.now - drag.range.start, drag.range.now - dx / (drag.box.width || 1) * (drag.range.now - drag.range.start));
    return true;
  }

  const activePage = () => view.page === "nodes" || view.page === "run";

  root.addEventListener("pointermove", (event) => {
    if (!activePage()) { pointerAt = null; return; }
    const surface = event.target.closest(".capacity-track, .activity-track, .chart-plot svg");
    if (!surface || ui.navigating || event.buttons) { pointerAt = null; return; }
    const box = surface.getBoundingClientRect(), range = getRange();
    pointerAt = range.start + fraction(event.clientX, box) * (range.now - range.start);
  });

  root.addEventListener("wheel", (event) => {
    if (!activePage()) return;
    const track = event.target.closest(SURFACES);
    if (!track || !(event.ctrlKey || event.metaKey) || (gesture && gesture.kind !== "wheel")) return;
    event.preventDefault();
    begin("wheel");
    const range = getRange(), at = fraction(event.clientX, track.getBoundingClientRect());
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? track.clientHeight : 1);
    const duration = clamp(Math.round((range.now - range.start) * Math.exp(clamp(delta * 0.002, -1, 1))), 60, MAX_WINDOW);
    select(duration, range.start + at * (range.now - range.start) + (1 - at) * duration);
    clearTimeout(settle);
    settle = setTimeout(() => finish(), 200);
  }, { passive: false });

  root.addEventListener("pointerdown", (event) => {
    if (!activePage()) return;
    const track = event.target.closest(SURFACES);
    if (!track || event.pointerType === "touch" || event.button !== 0 || event.target.closest("[data-placement], a")) return;
    finish();
    drag = { x: event.clientX, y: event.clientY, box: track.getBoundingClientRect(), range: getRange(), pointerId: event.pointerId };
  });
  window.addEventListener("pointermove", (event) => { if (drag?.pointerId === event.pointerId && moveDrag(event.clientX, event.clientY)) event.preventDefault(); });
  window.addEventListener("pointerup", (event) => { if (drag?.pointerId === event.pointerId) finish(); });
  window.addEventListener("pointercancel", (event) => { if (drag?.pointerId === event.pointerId) finish(true); });

  root.addEventListener("touchstart", (event) => {
    if (!activePage()) return;
    const track = event.target.closest(SURFACES);
    if (!track) return;
    if (event.touches.length === 2) {
      event.preventDefault();
      if (gesture?.kind === "wheel") finish();
      begin("pinch");
      gesture.kind = "pinch";
      drag = null;
      const [a, b] = event.touches;
      pinch = { box: track.getBoundingClientRect(), range: getRange(), x: (a.clientX + b.clientX) / 2, distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1 };
    } else if (event.touches.length === 1 && !event.target.closest("[data-placement], a")) {
      const touch = event.touches[0];
      drag = { x: touch.clientX, y: touch.clientY, box: track.getBoundingClientRect(), range: getRange() };
    }
  }, { passive: false });

  root.addEventListener("touchmove", (event) => {
    if (pinch && event.touches.length === 2) {
      event.preventDefault();
      const [a, b] = event.touches, x = (a.clientX + b.clientX) / 2;
      const duration = clamp(Math.round((pinch.range.now - pinch.range.start) * pinch.distance / (Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1)), 60, MAX_WINDOW);
      const anchor = pinch.range.start + fraction(pinch.x, pinch.box) * (pinch.range.now - pinch.range.start);
      select(duration, anchor + (1 - fraction(x, pinch.box)) * duration);
    } else if (drag && event.touches.length === 1 && moveDrag(event.touches[0].clientX, event.touches[0].clientY, true)) event.preventDefault();
  }, { passive: false });
  root.addEventListener("touchend", () => { if (drag || pinch) finish(); });
  root.addEventListener("touchcancel", () => { if (drag || pinch) finish(true); });

  root.addEventListener("click", (event) => {
    if (event.detail && performance.now() < ignoreClickUntil) {
      ignoreClickUntil = 0;
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, { capture: true });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && (gesture || drag || pinch)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      finish(true);
    }
  }, { capture: true });

  window.addEventListener("keydown", (event) => {
    if (!activePage()) return;
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d"].includes(key) || event.ctrlKey || event.metaKey || event.altKey || event.target.closest("input, select, textarea") || (gesture && gesture.kind !== "keys")) return;
    event.preventDefault();
    begin("keys");
    const range = getRange(), span = range.now - range.start;
    if (key === "a" || key === "d") {
      select(span, range.now + (key === "d" ? 0.2 : -0.2) * span);
      if (pointerAt !== null) {
        pointerAt += view.end - range.now;
        setCrosshair(pointerAt, getRange());
      }
    } else {
      const duration = clamp(Math.round(span * (key === "w" ? 0.8 : 1.25)), 60, MAX_WINDOW);
      const at = pointerAt === null ? 0.5 : clamp((pointerAt - range.start) / span, 0, 1);
      select(duration, range.start + at * span + (1 - at) * duration);
    }
    clearTimeout(settle);
    settle = setTimeout(() => finish(), 300);
  });

  window.addEventListener("blur", () => finish(true));
  window.addEventListener("hashchange", () => finish());
}
