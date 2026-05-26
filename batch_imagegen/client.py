from __future__ import annotations

from pixelbin import PixelbinClient, PixelbinConfig

DEFAULT_DOMAIN = "https://api.pixelbin.io"


def make_pixelbin_client(api_key: str, domain: str = DEFAULT_DOMAIN) -> PixelbinClient:
    """Create and return a configured PixelbinClient.

    Parameters
    ----------
    api_key:
        PixelBin API secret token (must be non-empty and at least 5 characters
        — the SDK enforces the minimum length at config construction time).
    domain:
        Base API domain.  Override in tests to point at a fake server.

    Raises
    ------
    ValueError
        If *api_key* is empty or blank.
    pixelbin.common.exceptions.PixelbinInvalidCredentialError
        If *api_key* is non-empty but shorter than 5 characters (SDK validates).
    """
    if not api_key or not api_key.strip():
        raise ValueError("PixelBin API key is required")
    return PixelbinClient(PixelbinConfig({"domain": domain, "apiSecret": api_key}))
