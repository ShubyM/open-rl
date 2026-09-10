// The Experiments page: recipe metrics read from each run's metrics.jsonl,
// rendered by views.experiments and charted per run here.

import { morph } from "./ui.js";
import { renderMetricChart } from "./charts.js";
import { experiments } from "./views.js";
import { ui, content, route } from "./store.js";
import { isFresh, metricData } from "./cache.js";

const URL = "/api/v1/dashboard/experiments";
const shortName = (name) => name.replace(/^gsm8k_rl_(mega|rank_sweep)_/, "");

export function experimentsPage() {
  morph(content, experiments(ui.experimentData));
  paintExperiments();
  if (isFresh(URL)) return;
  metricData(URL)
    .then((data) => {
      ui.experimentData = data;
      if (route()[0] !== "experiments") return;
      morph(content, experiments(data));
      paintExperiments();
    })
    .catch((error) => {
      if (route()[0] === "experiments" && !ui.experimentData) morph(content, experiments({ error: error.message, runs: [] }));
    });
}

function paintExperiments() {
  const data = ui.experimentData;
  if (!data) return;
  const byPath = new Map(data.runs.map((run) => [run.path, run]));
  content.querySelectorAll("[data-experiment]").forEach((element) => {
    const run = byPath.get(element.dataset.experiment);
    const points = run?.series[element.dataset.series] || [];
    if (!points.length) return;
    renderMetricChart(element, {
      samples: points,
      start: points[0][0],
      end: points.at(-1)[0],
      title: `${shortName(run.name)} · ${element.dataset.series}`,
      tone: "accent",
      xFormat: (step) => `step ${Math.round(step)}`,
    });
  });
}
