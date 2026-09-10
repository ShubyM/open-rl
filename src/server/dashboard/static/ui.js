export const escape = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const encode = encodeURIComponent;
export const empty = (text) => `<p class="empty-state">${escape(text)}</p>`;
export const button = (label, attrs = "") =>
  `<button type="button" class="appbutton" ${attrs}>${escape(label)}</button>`;
export const runStatus = (value) => {
  const label = String(value || "Unknown");
  const tone =
    {
      active: "running",
      running: "running",
      restarting: "pending",
      starting: "pending",
      "needs attention": "failed",
      queued: "pending",
      pending: "pending",
      failed: "failed",
      completed: "completed",
    }[label.toLowerCase()] || "unknown";
  return `<span class="run-state state-${tone}"><span aria-hidden="true"></span>${escape(label[0].toUpperCase() + label.slice(1))}</span>`;
};

export function elapsedTime(run, observedAt) {
  const timestamp = (value) => {
    if (value === null || value === undefined || value === "") return NaN;
    return typeof value === "number" ? value * 1000 : Date.parse(value);
  };
  const start = timestamp(run.created_at);
  const terminal = ["completed", "failed", "cancelled", "canceled"].includes(
    String(run.status || "").toLowerCase(),
  );
  const end = timestamp(terminal ? run.completed_at : observedAt);
  if (
    !Number.isFinite(start) ||
    start <= 0 ||
    !Number.isFinite(end) ||
    end < start
  )
    return "—";
  const minutes = Math.floor((end - start) / 60000);
  if (minutes < 1) return "<1m";
  return minutes < 60
    ? `${minutes}m`
    : `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}
