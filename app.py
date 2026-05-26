# app.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from batch_imagegen.paths import batches_dir, data_root, downloads_dir
from batch_imagegen.store import BatchStore
from batch_imagegen.orchestrator import OrchestratorRunner
from batch_imagegen.batch_builder import (
    parse_url_list, build_batch_from_inputs, ValidationError,
)
from batch_imagegen.models import JobStatus, TERMINAL_STATUSES
from batch_imagegen.zipper import build_zip
import pandas as pd

NANO_RATIOS = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
GPT_RATIOS  = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
RESOLUTIONS = ["1K", "2K", "4K"]


st.set_page_config(page_title="Batch Image Gen", layout="wide")


@st.cache_resource
def _store() -> BatchStore:
    return BatchStore(batches_dir())


@st.cache_resource
def _runner() -> OrchestratorRunner:
    return OrchestratorRunner(_store())


def _save_uploaded_files(files) -> list[str]:
    dest = data_root() / "uploads"
    dest.mkdir(exist_ok=True)
    paths: list[str] = []
    for uf in files:
        target = dest / uf.name
        target.write_bytes(uf.getbuffer())
        paths.append(str(target))
    return paths


def _render_new_batch() -> None:
    st.subheader("Create a new batch")
    name = st.text_input("Batch name", value="")
    model_label = st.radio("Model", ["Nano Banana Pro", "GPT Image 2"], horizontal=True)
    model = "nanoBananaPro_generate" if model_label == "Nano Banana Pro" else "gpt2_generate"

    with st.expander("Model parameters", expanded=True):
        if model == "nanoBananaPro_generate":
            aspect = st.selectbox("Aspect ratio", NANO_RATIOS, index=0)
            resolution = st.selectbox("Output resolution", RESOLUTIONS, index=0)
            params = {"aspect_ratio": aspect, "output_resolution": resolution}
        else:
            aspect = st.selectbox("Aspect ratio", GPT_RATIOS, index=0)
            resolution = st.selectbox("Output resolution", RESOLUTIONS, index=0)
            quality = st.selectbox("Quality", ["low", "medium", "high"], index=2)
            params = {"aspect_ratio": aspect, "output_resolution": resolution, "quality": quality}

    prompt = st.text_area("Prompt", height=150)
    st.caption(f"{len(prompt)} chars")

    # While submitting, lock the form so the user can't add more files or double-submit.
    is_submitting = st.session_state.get("submitting", False)

    source_mode = st.radio("Image source", ["Upload files", "Paste URLs"],
                           horizontal=True, disabled=is_submitting)
    sources: list[str] = []
    if source_mode == "Upload files":
        uploaded = st.file_uploader(
            "Drop images here", accept_multiple_files=True,
            type=["jpg", "jpeg", "png", "webp"],
            disabled=is_submitting,
        )
        if uploaded:
            sources = _save_uploaded_files(uploaded)
            st.caption(f"{len(uploaded)} files · "
                       f"~{sum(uf.size for uf in uploaded)/1_000_000:.1f} MB")
    else:
        text = st.text_area("One URL per line", height=120, key="url_list_text",
                            disabled=is_submitting)
        if text:
            try:
                sources = parse_url_list(text)
                st.caption(f"{len(sources)} URLs parsed")
            except ValidationError as e:
                st.error(str(e))

    concurrency = int(st.session_state.get("concurrency", 5))
    if sources:
        est_min = max(1, len(sources) / max(1, concurrency) * 30 / 60)
        st.caption(f"Estimated: ~{est_min:.0f} min at {concurrency} workers")

    api_key = st.session_state.get("api_key", "")
    disabled = is_submitting or not (name.strip() and prompt.strip() and sources and api_key)
    if is_submitting:
        tooltip = "Starting batch…"
        label = "Starting…"
    else:
        tooltip = "" if not disabled else "Fill name, prompt, sources, and add an API key in the sidebar."
        label = "Start batch"

    if st.button(label, disabled=disabled, help=tooltip, type="primary", key="start_batch"):
        # Mark submitting immediately so this branch can't be re-entered.
        st.session_state["submitting"] = True
        try:
            # Pre-flight: confirm all local-file sources are present on disk.
            # (Sources that are http(s) URLs are passed through unchanged.)
            missing = [
                s for s in sources
                if not (s.startswith("http://") or s.startswith("https://"))
                and not Path(s).exists()
            ]
            if missing:
                st.session_state["submitting"] = False
                st.error(
                    f"{len(missing)} uploaded file(s) could not be found on disk: "
                    + ", ".join(Path(m).name for m in missing[:5])
                    + ("…" if len(missing) > 5 else "")
                    + ". Please re-upload."
                )
                return

            with st.spinner("Starting batch…"):
                try:
                    batch = build_batch_from_inputs(
                        name=name, model=model, prompt=prompt,
                        params=params, concurrency=concurrency, sources=sources,
                    )
                except ValidationError as e:
                    st.session_state["submitting"] = False
                    st.error(str(e))
                    return
                _store().save(batch)
                _runner().submit(batch.batch_id, api_key)
            st.toast(f"Batch '{batch.name}' started — {len(batch.jobs)} images queued")
            st.session_state["submitting"] = False
            # Defer page switch — we cannot write to a widget-backed key
            # ("page") after the radio has been instantiated this run.
            # main() picks this up before rendering the sidebar on the next rerun.
            st.session_state["_pending_page"] = "Batches"
            st.session_state["selected_batch"] = batch.batch_id
            st.rerun()
        except Exception:
            st.session_state["submitting"] = False
            raise


def _counts(batch) -> dict[str, int]:
    out = {s.value: 0 for s in JobStatus}
    for j in batch.jobs:
        out[j.status.value] += 1
    return out


def _render_batches_page() -> None:
    batches = _store().list_batches()
    if not batches:
        st.info("No batches yet. Create your first batch in the sidebar → New batch.")
        return

    rows = []
    for b in batches:
        c = _counts(b)
        done = c["SUCCESS"] + c["FAILURE"]
        rows.append({
            "id": b.batch_id, "name": b.name, "model": b.model,
            "created": b.created_at, "progress": f"{done}/{len(b.jobs)}",
            "status": "completed" if b.completed_at else "running",
        })
    df = pd.DataFrame(rows)
    selected = st.session_state.get("selected_batch")

    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"id": None})

    pick = st.selectbox("Open batch", options=[b.batch_id for b in batches],
                        format_func=lambda i: next(b.name for b in batches if b.batch_id == i),
                        index=([b.batch_id for b in batches].index(selected)
                               if selected in [b.batch_id for b in batches] else 0))
    st.session_state["selected_batch"] = pick

    batch = next(b for b in batches if b.batch_id == pick)
    c = _counts(batch)
    done = c["SUCCESS"] + c["FAILURE"]
    st.progress(done / max(1, len(batch.jobs)),
                text=f"{c['PENDING']} pending · {c['SUBMITTED']+c['RUNNING']+c['UPLOADING']} running · "
                     f"{c['SUCCESS']} success · {c['FAILURE']} failed")

    # Narrower side-by-side buttons (instead of 50/50 columns).
    col1, col2, _spacer = st.columns([2, 2, 8])
    with col1:
        zip_disabled = c["SUCCESS"] == 0
        # Zip cache is keyed on batch + last-change so it invalidates on retry.
        zip_cache_key = f"zip_built::{batch.batch_id}::{batch.completed_at or batch.created_at}"
        zip_built = st.session_state.get(zip_cache_key, False)

        zip_path = downloads_dir() / f"{batch.batch_id}.zip"
        if zip_built and zip_path.exists():
            # Transform: the original "Download ZIP" slot is now "Save ZIP"
            st.download_button(
                "Save ZIP",
                data=zip_path.read_bytes(),
                file_name=f"{batch.name or batch.batch_id}.zip",
                mime="application/zip",
                key=f"save_zip_{batch.batch_id}",
            )
        else:
            if st.button("Download ZIP", disabled=zip_disabled,
                         help="No completed images yet." if zip_disabled else "",
                         key=f"build_zip_{batch.batch_id}"):
                with st.spinner("Building ZIP…"):
                    build_zip(batch, downloads_dir())
                st.session_state[zip_cache_key] = True
                st.rerun()
    with col2:
        api_key = st.session_state.get("api_key", "")
        retry_disabled = c["FAILURE"] == 0 or not api_key
        if st.button("Retry failed", disabled=retry_disabled,
                     key=f"retry_{batch.batch_id}"):
            for j in batch.jobs:
                if j.status == JobStatus.FAILURE:
                    j.status = JobStatus.PENDING
                    j.attempts = 0
                    j.error = None
            batch.completed_at = None
            _store().save(batch)
            # Invalidate any cached zip flag since a retry will produce new outputs.
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith(f"zip_built::{batch.batch_id}::"):
                    del st.session_state[k]
            _runner().submit(batch.batch_id, api_key)
            st.toast("Retrying failed jobs…")
            st.rerun()

    with st.expander("Jobs"):
        jobs_df = pd.DataFrame([
            {
                "source": j.source.split("/")[-1],
                "status": j.status.value,
                "output": j.output_url or "",
                "error": (j.error or "")[:80],
            }
            for j in batch.jobs
        ])
        st.dataframe(jobs_df, hide_index=True, use_container_width=True)

    # Completion toast (fires once per batch_id per session)
    if batch.completed_at:
        last = st.session_state.setdefault("seen_completion", {})
        if last.get(batch.batch_id) != batch.completed_at:
            st.toast(f"✓ Batch '{batch.name}' done — "
                     f"{c['SUCCESS']} succeeded, {c['FAILURE']} failed")
            last[batch.batch_id] = batch.completed_at


def _sidebar() -> None:
    st.sidebar.header("Settings")
    st.sidebar.text_input(
        "PixelBin API Key", type="password", key="api_key",
        help="Stored only in this session. Never written to disk.",
    )
    st.sidebar.slider("Concurrent workers", 1, 20, value=5, key="concurrency")
    if not st.session_state.get("api_key"):
        st.sidebar.warning(
            "Paste your PixelBin API key to start. Get one at "
            "app.pixelbin.io › Settings › API tokens."
        )


def main() -> None:
    # Apply any deferred page switch BEFORE the radio widget is instantiated.
    # Streamlit forbids writing to a widget-backed session_state key after
    # the widget exists in the current run.
    if "_pending_page" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_pending_page")
    _sidebar()
    page = st.sidebar.radio("Page", ["New batch", "Batches"], key="page")
    st_autorefresh(interval=2000, key="auto_refresh")
    if page == "New batch":
        _render_new_batch()
    else:
        st.title("Batches")
        _render_batches_page()


if __name__ == "__main__":
    main()
