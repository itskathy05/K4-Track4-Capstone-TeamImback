from evaluate import expand


def test_evaluate_expands_globs(tmp_path):
    for name in ("fall-01-cam0.mp4", "adl-01-cam0.mp4"):
        (tmp_path / name).write_bytes(b"")
    found = expand([str(tmp_path / "*.mp4")])
    assert [p.split("\\")[-1].split("/")[-1] for p in found] == ["adl-01-cam0.mp4", "fall-01-cam0.mp4"]
