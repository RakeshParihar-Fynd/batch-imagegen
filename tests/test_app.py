# tests/test_app.py
from streamlit.testing.v1 import AppTest


def test_start_batch_disabled_without_api_key():
    at = AppTest.from_file("app.py", default_timeout=10).run()
    # Find the "Start batch" button
    start = next((b for b in at.button if b.label == "Start batch"), None)
    assert start is not None
    assert start.disabled is True


def test_empty_state_on_batches_page():
    at = AppTest.from_file("app.py", default_timeout=10).run()
    at.sidebar.radio[0].set_value("Batches").run()
    # "No batches yet." message should appear via st.info
    infos = [i for i in at.info if "No batches yet" in str(i.value)]
    assert infos, "Expected empty-state info message on Batches page"
