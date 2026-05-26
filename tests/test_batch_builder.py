from batch_imagegen.batch_builder import (
    parse_url_list, build_batch_from_inputs, ValidationError,
)
import pytest


def test_parse_url_list_handles_blanks_and_dupes():
    text = "https://a.com/x.jpg\n\n  https://b.com/y.jpg \nhttps://a.com/x.jpg"
    urls = parse_url_list(text)
    assert urls == ["https://a.com/x.jpg", "https://b.com/y.jpg"]


def test_parse_url_list_rejects_invalid():
    with pytest.raises(ValidationError) as exc:
        parse_url_list("https://a.com/ok.jpg\nnot-a-url\nftp://x")
    assert "lines 2, 3" in str(exc.value)


def test_build_batch_from_inputs_minimal():
    batch = build_batch_from_inputs(
        name="My batch",
        model="nanoBananaPro_generate",
        prompt="hello",
        params={"aspect_ratio": "auto", "output_resolution": "1K"},
        concurrency=4,
        sources=["https://a.com/x.jpg", "https://b.com/y.jpg"],
    )
    assert batch.name == "My batch"
    assert len(batch.jobs) == 2
    assert batch.concurrency == 4
    assert all(j.source.startswith("https://") for j in batch.jobs)


def test_build_batch_requires_prompt_and_sources():
    with pytest.raises(ValidationError):
        build_batch_from_inputs(
            name="x", model="nanoBananaPro_generate", prompt="",
            params={}, concurrency=1, sources=["https://a/x"],
        )
    with pytest.raises(ValidationError):
        build_batch_from_inputs(
            name="x", model="nanoBananaPro_generate", prompt="p",
            params={}, concurrency=1, sources=[],
        )
