import subprocess
import time
import urllib.request
import os
import signal

import pytest
import importlib


def _wait_for_url(url, timeout=15.0):
    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.read().decode("utf-8")
        except Exception:
            if time.time() - start > timeout:
                raise
            time.sleep(0.2)


def test_streamlit_app_smoke(tmp_path):
    """Start the Streamlit app in a subprocess and check the main page loads.

    This is a lightweight end-to-end smoke test that doesn't require a browser driver.
    """
    port = 8502
    env = os.environ.copy()
    # run streamlit headless on a non-default port
    proc = subprocess.Popen([
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    try:
        body = _wait_for_url(f"http://localhost:{port}", timeout=20.0)
        # Streamlit serves an initial HTML shell (client-side rendered).
        # Smoke-check that the server responds with HTML (status 200) rather than checking client-rendered text.
        assert "<html" in body.lower() or "<!doctype html" in body.lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


@pytest.mark.skipif(os.getenv("RUN_PLAYWRIGHT") != "1", reason="Playwright E2E disabled by default")
def test_streamlit_ui_playwright(page):
    """Optional Playwright UI test — enabled by setting RUN_PLAYWRIGHT=1 in the environment."""
    # start server on port 8503
    port = 8503
    proc = subprocess.Popen([
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # wait for server
        _wait_for_url(f"http://localhost:{port}", timeout=20.0)
        url = f"http://localhost:{port}"
        page.goto(url)
        # click the run button in the sidebar
        page.click("text=Run simulation & train model")
        # wait for metrics text to appear
        page.wait_for_selector("text=Regression metrics", timeout=20000)
        assert page.is_visible("text=Regression metrics")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()