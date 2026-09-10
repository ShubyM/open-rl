import { escape } from "./ui.js";

const nodeTimePresets = [
  [300, "Last 5 minutes"],
  [900, "Last 15 minutes"],
  [1800, "Last 30 minutes"],
  [3600, "Last 1 hour"],
  [21600, "Last 6 hours"],
  [86400, "Last 24 hours"],
];
const nodeTimeBindings = new WeakMap();

export function resolveTimeRange(selection, nowSeconds) {
  const requestedDuration = Number(selection?.duration);
  const duration =
    Number.isFinite(requestedDuration) && requestedDuration > 0
      ? Math.min(requestedDuration, 86400)
      : 1800;
  const requestedEnd = selection?.end;
  const end =
    requestedEnd != null && Number.isFinite(Number(requestedEnd))
      ? Math.min(Number(requestedEnd), nowSeconds)
      : nowSeconds;
  return { start: end - duration, end };
}

function nodeTimeInput(seconds) {
  return new Date(seconds * 1000).toISOString().slice(0, 16);
}

function nodeTimeLabel(selection, range) {
  if (selection.end == null) {
    const preset = nodeTimePresets.find(
      ([duration]) => duration === selection.duration,
    );
    if (preset) return preset[1];
    const minutes = Math.round((range.end - range.start) / 60);
    return `Last ${minutes} minute${minutes === 1 ? "" : "s"}`;
  }
  const dateFormat = new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
  const start = new Date(range.start * 1000);
  const end = new Date(range.end * 1000);
  const startTime = start.toISOString().slice(11, 16);
  const endTime = end.toISOString().slice(11, 16);
  const endDate =
    start.toISOString().slice(0, 10) === end.toISOString().slice(0, 10)
      ? ""
      : `${dateFormat.format(end)}, `;
  return `${dateFormat.format(start)}, ${startTime} – ${endDate}${endTime} UTC`;
}

export function timeRangeControl(selection, nowSeconds) {
  const range = resolveTimeRange(selection, nowSeconds);
  const rolling = selection.end == null;
  const label = nodeTimeLabel(selection, range);
  const title = `${new Date(range.start * 1000).toISOString()} – ${new Date(range.end * 1000).toISOString()}`;
  return `<div class="node-time-controls" role="group" aria-label="Node time range">
    <div class="node-time-picker">
      <div class="node-time-navigation">
        <button type="button" class="appbutton" data-node-time="previous" aria-label="Previous time range" title="Previous time range"><span aria-hidden="true">‹</span></button>
        <button type="button" class="appbutton node-time-toggle" data-node-time="toggle" aria-expanded="false" aria-controls="node-time-popover" title="${escape(title)}"><span>${escape(label)}</span><svg class="node-time-chevron" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
        <button type="button" class="appbutton" data-node-time="next" aria-label="Next time range" title="Next time range" ${rolling || range.end >= nowSeconds ? "disabled" : ""}><span aria-hidden="true">›</span></button>
      </div>
      <div id="node-time-popover" class="node-time-popover" hidden>
        <div class="node-time-presets" role="group" aria-label="Quick time ranges">${nodeTimePresets.map(([duration, text]) => `<button type="button" class="appbutton" data-node-time-preset="${duration}" aria-pressed="${rolling && selection.duration === duration}">${text}</button>`).join("")}</div>
        <form class="node-time-custom" data-node-time-form novalidate>
          <div class="node-time-custom-heading"><strong>Custom range</strong><span>UTC · up to 24 hours</span></div>
          <label>From (UTC)<input type="datetime-local" name="node-time-from" required value="${nodeTimeInput(range.start)}" max="${nodeTimeInput(nowSeconds)}"></label>
          <label>To (UTC)<input type="datetime-local" name="node-time-to" required value="${nodeTimeInput(range.end)}" max="${nodeTimeInput(nowSeconds)}"></label>
          <p class="node-time-error" data-node-time-error role="alert" hidden></p>
          <button type="button" class="appbutton node-time-apply" data-node-time="apply">Apply range</button>
        </form>
      </div>
    </div>
  </div>`;
}

export function bindTimeRange(
  root,
  getSelection,
  onChange,
  getNow = () => Date.now() / 1000,
) {
  const existing = nodeTimeBindings.get(root);
  if (existing) {
    Object.assign(existing, { getSelection, onChange, getNow });
    return;
  }
  const binding = { getSelection, onChange, getNow };
  nodeTimeBindings.set(root, binding);
  const picker = () => root.querySelector(".node-time-picker");
  const alignPopover = () => {
    const current = picker();
    const popover = current?.querySelector(".node-time-popover");
    if (!popover || popover.hidden) return;
    popover.style.left = "";
    popover.style.right = "";
    const bounds = popover.getBoundingClientRect();
    const owner = root.getBoundingClientRect();
    const left = Math.max(
      owner.left + 14,
      Math.min(bounds.left, owner.right - 14 - bounds.width),
    );
    popover.style.left = `${left - current.getBoundingClientRect().left}px`;
    popover.style.right = "auto";
  };
  root.ownerDocument.defaultView.addEventListener("resize", alignPopover);
  const close = (restoreFocus = false) => {
    const current = picker();
    if (!current) return;
    const toggle = current.querySelector('[data-node-time="toggle"]');
    current.querySelector(".node-time-popover").hidden = true;
    toggle.setAttribute("aria-expanded", "false");
    if (restoreFocus) toggle.focus();
  };
  const change = async (selection) => {
    close();
    try {
      await binding.onChange(selection);
    } finally {
      picker()?.querySelector('[data-node-time="toggle"]')?.focus();
    }
  };
  root.addEventListener("click", (event) => {
    const target = event.target.closest(
      "[data-node-time], [data-node-time-preset]",
    );
    if (!target || !root.contains(target) || target.disabled) return;
    const now = binding.getNow();
    const selection = binding.getSelection();
    const range = resolveTimeRange(selection, now);
    const duration = range.end - range.start;
    const action = target.dataset.nodeTime;
    if (action === "toggle") {
      const popover = picker().querySelector(".node-time-popover");
      const opening = popover.hidden;
      popover.hidden = !opening;
      target.setAttribute("aria-expanded", String(opening));
      if (opening) {
        alignPopover();
        const form = popover.querySelector("form");
        form.elements["node-time-from"].value = nodeTimeInput(range.start);
        form.elements["node-time-to"].value = nodeTimeInput(range.end);
        form.elements["node-time-from"].max = nodeTimeInput(now);
        form.elements["node-time-to"].max = nodeTimeInput(now);
        form.querySelector("[data-node-time-error]").hidden = true;
        popover
          .querySelector('[aria-pressed="true"], [data-node-time-preset]')
          .focus();
      }
    } else if (action === "apply") {
      applyCustom(picker().querySelector("form"));
    } else if (target.dataset.nodeTimePreset) {
      void change({
        duration: Number(target.dataset.nodeTimePreset),
        end: null,
      });
    } else if (action === "previous") {
      void change({ duration, end: range.start });
    } else if (action === "next") {
      const end = Math.min(range.end + duration, now);
      void change({ duration, end: end >= now ? null : end });
    }
  });
  const applyCustom = (form) => {
    const start =
      Date.parse(`${form.elements["node-time-from"].value}Z`) / 1000;
    const end = Date.parse(`${form.elements["node-time-to"].value}Z`) / 1000;
    let message = "";
    if (!Number.isFinite(start) || !Number.isFinite(end))
      message = "Enter both dates and times.";
    else if (end <= start) message = "To must be later than From.";
    else if (end > binding.getNow())
      message = "Choose a range that ends at or before the current time.";
    else if (end - start > 86400)
      message = "Choose a range of 24 hours or less.";
    const error = form.querySelector("[data-node-time-error]");
    error.textContent = message;
    error.hidden = !message;
    if (!message) void change({ duration: end - start, end });
  };
  root.addEventListener("submit", (event) => {
    if (!event.target.matches("[data-node-time-form]")) return;
    event.preventDefault();
    applyCustom(event.target);
  });
  root.addEventListener("keydown", (event) => {
    if (
      event.key === "Enter" &&
      event.target.matches("[data-node-time-form] input")
    ) {
      event.preventDefault();
      applyCustom(event.target.closest("form"));
    }
    if (
      event.key === "Escape" &&
      picker()?.querySelector(".node-time-popover")?.hidden === false
    ) {
      event.preventDefault();
      close(true);
    }
  });
  root.ownerDocument.addEventListener("click", (event) => {
    const current = picker();
    if (current && !current.contains(event.target)) close();
  });
}
