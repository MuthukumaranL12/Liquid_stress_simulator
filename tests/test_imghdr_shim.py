from imghdr import what


def test_imghdr_shim_png_signature():
    png_sig = b"\x89PNG\r\n\x1a\n" + b"rest"
    assert what(png_sig) == "png"


def test_imghdr_shim_jpeg_signature():
    jpeg_sig = b"\xff\xd8\xff\xe0" + b"rest"
    assert what(jpeg_sig) == "jpeg"


def test_imghdr_shim_unknown():
    assert what(b"notanimage") is None
