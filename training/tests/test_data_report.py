from data_report import count_images, report


def test_count_images_missing_folder_is_zero(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "1.png").write_bytes(b"x")
    (tmp_path / "A" / "2.jpg").write_bytes(b"x")
    (tmp_path / "A" / "notes.txt").write_text("not an image")
    (tmp_path / "B").mkdir()
    assert count_images(str(tmp_path), ["A", "B", "C"]) == {"A": 2, "B": 0, "C": 0}


def test_report_fails_on_empty_class():
    code, lines = report({"A": 12, "B": 0, "C": 0})
    assert code == 1
    text = "\n".join(lines)
    assert "B" in text and "C" in text


def test_report_lists_thin_classes():
    code, lines = report({"A": 12, "B": 5})
    assert code == 0
    assert any("B (5)" in line for line in lines)
