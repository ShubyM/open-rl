// Failed fields keep the last samples for the same physical GPU or worker process.
// Successful empty fields are authoritative and clear old samples.
export function mergeSeries(next, previous) {
  if (!previous) return next;
  const retain = (row, old, errors) => {
    const retained = { ...row };
    for (const key in errors) if (old?.[key]) retained[key] = old[key];
    return retained;
  };
  const rows = (name, identity) => {
    const old = new Map((previous[name] || []).map((row) => [identity(row), row]));
    return (next[name] || []).map((row) => retain(row, old.get(identity(row)), row.errors));
  };
  return {
    ...next,
    devices: rows("devices", (row) => JSON.stringify([row.id, row.uuid])),
    workers: rows("workers", (row) => JSON.stringify([row.id, row.pod_uid, row.pod])),
    ...(next.samplers ? { samplers: retain(next.samplers, previous.samplers, next.samplers_errors) } : {}),
  };
}
export const seriesError = (rows, field, error) => [...new Set([error, ...rows.map((row) => row.errors?.[field])].filter(Boolean))].join(" · ");
