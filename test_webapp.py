import json
import math
import threading
import urllib.error
import urllib.request

import pytest

from fall_detector import Config
from webapp import list_sources, make_server, parse_results, resolve_source, safe_upload_name, update_config


def test_webcam_source_resolves_without_crop():
    path, crop, label = resolve_source("cam:0")
    assert (path, crop) == ("0", False) and "Webcam" in label


def test_dataset_clips_are_listed_with_activity_labels_and_cropped():
    clips = [s for s in list_sources() if s["group"] != "Webcam"]
    if not clips:
        pytest.skip("run download_data.py first")
    fall = next(s for s in clips if "fall-01" in s["id"])
    assert "fall" in fall["label"].lower()
    path, crop, _ = resolve_source(fall["id"])
    assert crop is True and path.endswith("fall-01-cam0.mp4")


def test_unknown_or_traversal_sources_are_rejected():
    for bad in ("../README.md", "C:/Windows/win.ini", "test_videos/../main.py", "cam:abc", ""):
        with pytest.raises(ValueError):
            resolve_source(bad)


def test_update_config_coerces_numbers_and_rejects_bad_input():
    cfg = Config()
    update_config(cfg, {"fall_speed": "2.5"})
    assert cfg.fall_speed == 2.5
    for bad in ({"nope": 1}, {"fall_speed": -1}, {"fall_speed": math.nan}, {"fall_speed": "x"},
                {"smooth_alpha": 0}, {"horizontal_angle": 95}):
        with pytest.raises(ValueError):
            update_config(cfg, bad)
    assert cfg.fall_speed == 2.5  # rejected updates change nothing


def test_safe_upload_name_strips_directories_and_checks_extension():
    assert safe_upload_name("../../my clip.MP4") == "my_clip.MP4"
    with pytest.raises(ValueError):
        safe_upload_name("evil.exe")


def test_parse_results_reads_markdown_table():
    md = "| clip | expected | alerted | first alert (s) | pose found | result |\n|---|---|---|---|---|---|\n" \
         "| fall-01 | alert | yes | 4.8 | 76% | OK |\n\n9/10 clips as expected\n"
    assert parse_results(md) == [{"clip": "fall-01", "expected": "alert", "alerted": "yes",
                                  "first alert (s)": "4.8", "pose found": "76%", "result": "OK"}]


def test_server_serves_page_sources_and_validates_config():
    server = make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"<html" in urllib.request.urlopen(base + "/").read().lower()
        sources = json.loads(urllib.request.urlopen(base + "/api/sources").read())
        assert sources[0]["id"] == "cam:0"
        req = urllib.request.Request(base + "/api/config", data=b'{"fall_speed": -3}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 400
        req = urllib.request.Request(base + "/api/start", data=b'{"source": "../main.py"}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
