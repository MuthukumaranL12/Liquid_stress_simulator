"""Minimal `imghdr` shim for environments where stdlib `imghdr` is missing.

Provides `what(file, h=None)` used by Streamlit to detect basic image formats.
This is a small compatibility layer — prefer using the stdlib `imghdr` when
available.
"""
from typing import Optional


def _startswith(data: bytes, prefix: bytes) -> bool:
    return data[: len(prefix)] == prefix


def what(file, h: Optional[bytes] = None) -> Optional[str]:
    """Detect image type. Accepts filename/path or a file-like object or raw header bytes.

    Returns one of: 'jpeg','png','gif','bmp','tiff','webp','ico' or None.
    """
    # allow callers to pass raw header bytes as the `file` argument (matches CPython `imghdr.what` behaviour when used in tests)
    if isinstance(file, (bytes, bytearray)) and h is None:
        header = bytes(file)
    else:
        header = h

    if header is None:
        # If a filename (str / os.PathLike), read its header; or accept a file-like object
        try:
            if isinstance(file, (str, bytes)):
                # bytes-as-filename handled above; this branch is for str paths
                with open(file, "rb") as f:
                    header = f.read(32)
            else:
                # file-like object with read()
                header = file.read(32)
                # rewind if possible
                try:
                    file.seek(0)
                except Exception:
                    pass
        except Exception:
            return None

    if not header:
        return None

    b = header
    # JPEG (starts with FF D8)
    if _startswith(b, b"\xff\xd8\xff"):
        return "jpeg"
    # PNG
    if _startswith(b, b"\x89PNG\r\n\x1a\n"):
        return "png"
    # GIF
    if _startswith(b, b"GIF87a") or _startswith(b, b"GIF89a"):
        return "gif"
    # BMP
    if _startswith(b, b"BM"):
        return "bmp"
    # TIFF (II or MM)
    if _startswith(b, b"II\x2a\x00") or _startswith(b, b"MM\x00\x2a"):
        return "tiff"
    # WEBP (RIFF....WEBP)
    if _startswith(b, b"RIFF") and b[8:12] == b"WEBP":
        return "webp"
    # ICO (starts with 00 00 01 00)
    if _startswith(b, b"\x00\x00\x01\x00"):
        return "ico"

    return None
