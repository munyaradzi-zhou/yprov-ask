from __future__ import annotations

import importlib
import os

import yprov_ask


def test_a_key_log_on_a_windows_device_path_is_dropped(monkeypatch):
    # Windows stats \.\pipe-like paths as regular files, so the name alone must decide.
    monkeypatch.setenv("SSLKEYLOGFILE", r"\.\aswMonFltProxy\267201e954a67f")
    importlib.reload(yprov_ask)
    assert "SSLKEYLOGFILE" not in os.environ


def test_a_real_key_log_file_is_kept(monkeypatch, tmp_path):
    keylog = tmp_path / "keys.log"
    keylog.write_text("")
    monkeypatch.setenv("SSLKEYLOGFILE", str(keylog))
    importlib.reload(yprov_ask)
    assert os.environ["SSLKEYLOGFILE"] == str(keylog)
