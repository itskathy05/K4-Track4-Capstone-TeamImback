from evaluate import expand


def test_evaluate_expands_globs(tmp_path):
    for name in ("fall-01-cam0.mp4", "adl-01-cam0.mp4"):
        (tmp_path / name).write_bytes(b"")
    found = expand([str(tmp_path / "*.mp4")])
    assert [p.split("\\")[-1].split("/")[-1] for p in found] == ["adl-01-cam0.mp4", "fall-01-cam0.mp4"]


def test_evaluate_warns_about_patterns_that_match_nothing(tmp_path, capsys):
    (tmp_path / "fall-01-cam0.mp4").write_bytes(b"")
    found = expand([str(tmp_path / "*.mp4"), str(tmp_path / "typo*.mp4")])
    assert len(found) == 1
    assert "typo" in capsys.readouterr().err


def test_score_clip_rows_match_results_table_columns():
    from evaluate import score_clip
    row = score_clip("fall-01", [(0.0, "NORMAL", True), (1.5, "FALL_ALERT", False)])
    assert row == {"clip": "fall-01", "expected": "alert", "alerted": "yes", "first alert (s)": "1.5",
                   "pose found": "50%", "result": "OK"}
    assert score_clip("adl-01", [(0.0, "FALL_ALERT", True)])["result"] == "MISS"
