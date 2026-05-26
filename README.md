# batch-imagegen

Local Streamlit tool for the annotation team to run batch image generation jobs against the [PixelBin Prediction API](https://www.pixelbin.io/docs/prediction-api/overview). Supports **Nano Banana Pro** and **GPT Image 2**.

## Features

- Drop in 100+ images (local files or URLs), one prompt, one model
- Worker pool with configurable concurrency (default 5)
- Live progress with auto-refresh
- Retry failed jobs
- Download all successful outputs as a ZIP with a `manifest.csv`
- All state on disk under `~/.batch-imagegen/` — survives reloads

## Setup

```bash
pip install -e ".[dev]"
```

## Run

```bash
streamlit run app.py
```

Open the URL it prints. In the sidebar:

1. Paste your PixelBin API token (Settings → API tokens in the PixelBin console). It is stored only in this browser session — never written to disk.
2. Adjust "Concurrent workers" if needed.

Then on the **New batch** page, fill name + model + prompt, pick "Upload files" or "Paste URLs", and click **Start batch**. The **Batches** page shows live progress and the **Download ZIP** button.

## Where things live

```
~/.batch-imagegen/
├── batches/<batch_id>.json     # one file per batch
├── uploads/                    # local files copied here before upload
└── downloads/<batch_id>.zip    # built on demand
```

Override the root with `BATCH_IMAGEGEN_HOME=/some/path streamlit run app.py`.

## Tests

```bash
pytest -v
```

The E2E test against a fake server is gated by `RUN_E2E=1`. See `tests/test_e2e_fake.py`.

## Caveats

- Streamlit's hot-reloader kills the background orchestrator thread on code changes. Restart `streamlit run` if jobs stall.
- The tool processes one batch at a time. Submitting a second batch while one is running queues it.
- No cancellation in v1 — once started, a batch runs to completion.

## Manual smoke checklist

Before each release:

- [ ] Run `pytest` — all green
- [ ] `streamlit run app.py`, paste a real test API key, kick off a 5-image batch with 2 known-bad URLs — verify 3 successes + 2 failures + download ZIP contains 3 images + manifest with all 5
- [ ] Refresh the browser mid-batch — progress should resume from JSON
- [ ] Hit **Retry failed** — verify those 2 re-run
