# Harvey LAB Trace Collection

Collect teacher traces from Gemini Flash on Harvey LAB for gemma-4-E4B SFT/RL. The toolkit keeps only attempts that pass all Harvey LAB scoring checks and stores the full patched transcript.

## Setup

```bash
cd experiments/lab-traces
./setup.sh
```

`GOOGLE_API_KEY` is needed for Gemini runs. `setup.sh` clones `harvey-labs/`, checks out the pinned SHA, applies `patches/full-transcript.patch`, and runs `uv sync`.

## Split

```bash
python3 make_split.py --areas banking-finance --areas insurance --out split.json
```

## Collect

Single task pilot:

```bash
./harvey-labs/.venv/bin/python collect.py --task banking-finance/identify-issues-in-commitment-letter --max-attempts 2 --parallel 1
```

Split run:

```bash
./harvey-labs/.venv/bin/python collect.py --split split.json --subset train --model gemini-3.5-flash --judge-model gemini-3.5-flash --parallel 1
```

Gemini 3.5 Flash has a 2M input tokens/min paid-tier limit. Keep `--parallel` low
(1-2) until compaction lands.

## Output

Kept traces are written to `traces/<task-id>/` where slashes in the task ID are replaced with `__`:

- `full_transcript.jsonl`
- `config.json`
- `metrics.json`
- `scores.json`
- `meta.json`

Collection status is appended to `traces/manifest.jsonl`. Existing kept tasks in the manifest are skipped on resume.

`harvey-labs/`, `traces/`, and `split.json` are gitignored.
