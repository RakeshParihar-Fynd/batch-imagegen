# PixelBin SDK surface check — 2026-05-25

Installed version: **4.3.0** (`pip show pixelbin`)

## Top-level client attributes

`PixelbinClient` exposes: `assets`, `config`, `organization`, `predictions`, `transformation`, `uploader`

---

## Predictions (used by predictor.py)

All methods are on `client.predictions`:

| Method | Signature | Notes |
|---|---|---|
| `create` | `(name: str, input: Dict[str,Any]=None, webhook: str|None=None) -> Dict[str,Any]` | Sync wrapper over `createAsync` |
| `createAsync` | `(name: str, input: Dict[str,Any]=None, webhook: str|None=None) -> Dict[str,Any]` | |
| `get` | `(request_id: str) -> Dict[str,Any]` | |
| `getAsync` | `(request_id: str) -> Dict[str,Any]` | |
| `wait` | `(request_id: str, options: Dict[str,Any]|None=None) -> Dict[str,Any]` | Polls to terminal state |
| `waitAsync` | `(request_id: str, options: Dict[str,Any]|None=None) -> Dict[str,Any]` | |
| `create_and_wait` | `(name: str, input: Dict[str,Any]=None, webhook: str|None=None, options: Dict[str,Any]|None=None) -> Dict[str,Any]` | Single call: create + poll |
| `create_and_waitAsync` | `(name: str, input: Dict[str,Any]=None, webhook: str|None=None, options: Dict[str,Any]|None=None) -> Dict[str,Any]` | |
| `list` | `() -> List[Dict[str,Any]]` | |
| `listAsync` | same | |
| `get_schema` | `(name: str) -> Dict[str,Any]` | |
| `get_schemaAsync` | same | |

**No `getAsync` signature difference** — both sync and async variants exist for every method.

---

## File upload — chosen method

### `client.uploader.upload` (recommended for `uploader.py`)

**Full signature:**
```python
client.uploader.upload(
    file: bytes | io.BufferedIOBase,
    name: str = None,
    path: str = None,
    format: str = None,
    access: AccessEnum = None,
    tags: List[str] = None,
    metadata: Any = None,
    overwrite: bool = None,
    filenameOverride: bool = None,
    expiry: int = None,
    uploadOptions: dict = {
        "chunkSize": 10485760,   # 10 MB
        "maxRetries": 2,
        "concurrency": 3,
        "exponentialFactor": 2,
    },
) -> dict
```

**Async variant:** `client.uploader.uploadAsync(...)` — identical signature, returns `dict`.

**How it works internally:**
1. Calls `client.assets.createSignedUrlV2Async(...)` to obtain a pre-signed multipart URL.
2. Chunked-uploads the file to S3 (each part → `PUT {url}&partNumber=N`).
3. Completes the multipart upload with `POST {url}` carrying `{"parts": [1…N], ...fields}`.
4. Returns the JSON body of the completion response.

**Return value shape** — defined by `UploadResponse` marshmallow schema:

```python
{
    "_id":       str,       # internal asset ID
    "fileId":    str,
    "name":      str,
    "path":      str,
    "format":    str,
    "size":      int,
    "access":    str,       # "public-read" | "private"
    "tags":      List[str],
    "metadata":  dict,
    "url":       str,       # <-- CDN URL (top-level "url" key)
    "thumbnail": str,
}
```

**CDN URL field: `response["url"]`** (top-level key, NOT `response["cdn"]["url"]`).

### Alternative: `client.assets.fileUpload`

```python
client.assets.fileUpload(
    file: FileIO = None,     # requires io.FileIO — NOT bytes
    path: str = None,
    name: str = None,
    access: AccessEnum = None,
    tags: List[str] = None,
    metadata: Any = None,
    overwrite: bool = None,
    filenameOverride: bool = None,
) -> dict
```

Returns the same `UploadResponse` shape (same `url` key).

**Why `uploader.upload` is preferred:** accepts `bytes | BufferedIOBase` (more flexible than `FileIO`), supports multipart chunked upload with retry/concurrency options, and has an `async` variant without the `asyncio.get_event_loop()` hack.

---

## Notes / caveats

1. **`PixelbinConfig` minimum key length is 5 characters.** Passing an API key shorter than 5 chars raises `PixelbinInvalidCredentialError` at config construction time, before any network call. The `make_pixelbin_client` factory's `ValueError` guard catches empty strings; the SDK itself guards against too-short strings.

2. **`asyncio.get_event_loop()` in sync wrappers.** Both `assets.fileUpload` and `uploader.upload` call `asyncio.get_event_loop().run_until_complete(...)`. This works in ordinary script/thread contexts but will fail if called from within a running event loop (e.g., inside an `async def`). In `uploader.py`, call `uploadAsync` directly from async context.

3. **`client.files` does NOT exist** — the attribute is `assets` (contains `fileUpload`, `fileUploadAsync`, `urlUpload`, etc.).

4. **Predictions `wait` / `create_and_wait` poll internally** — no need to implement polling manually in `predictor.py`.
