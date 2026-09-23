# tests/test_check_no_secrets.py
from pathlib import Path
import pytest
from scripts.check_no_secrets import scan_text, scan_repo, load_deny_terms


def test_flags_hf_token():
    findings = scan_text("token = 'hf_" + "a" * 34 + "'", [])
    assert [f.kind for f in findings] == ["token"]


def test_flags_windows_absolute_path():
    findings = scan_text(r"p = 'C:\Users\someone\thing'", [])
    assert [f.kind for f in findings] == ["abs_path"]


def test_flags_posix_home_path():
    findings = scan_text("p = '/home/someone/thing'", [])
    assert [f.kind for f in findings] == ["abs_path"]


def test_flags_deny_term_case_insensitively():
    findings = scan_text("we use SecretProject here", ["secretproject"])
    assert [f.kind for f in findings] == ["deny_term"]


def test_clean_text_yields_nothing():
    findings = scan_text("model_id = 'Qwen/Qwen3-8B'", ["secretproject"])
    assert findings == []


def test_reports_line_numbers():
    findings = scan_text("ok\nok\n/home/someone\n", [])
    assert findings[0].line_no == 3


def test_missing_deny_file_is_not_an_error(tmp_path):
    assert load_deny_terms(tmp_path) == []


def test_deny_file_ignores_blanks_and_comments(tmp_path):
    (tmp_path / ".secretscan-deny").write_text(
        "# a comment\n\nalpha\nbeta\n", encoding="utf-8"
    )
    assert load_deny_terms(tmp_path) == ["alpha", "beta"]


def test_scan_repo_skips_gitignored_and_binary(tmp_path):
    (tmp_path / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "leak.py").write_text("/home/someone", encoding="utf-8")
    (tmp_path / "clean.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "img.png").write_bytes(b"\x89PNG\x00/home/someone")
    assert scan_repo(tmp_path, []) == []


def test_scan_repo_finds_a_real_leak(tmp_path):
    (tmp_path / "bad.py").write_text("/home/someone/x", encoding="utf-8")
    results = scan_repo(tmp_path, [])
    assert len(results) == 1 and results[0][0].name == "bad.py"


def test_common_words_do_not_blind_the_scanner():
    findings = scan_text("findings: hf_" + "a" * 34, [])
    assert [f.kind for f in findings] == ["token"]


def test_tmp_path_does_not_blind_the_scanner():
    findings = scan_text("tmp_path = '/home/someone/x'", [])
    assert [f.kind for f in findings] == ["abs_path"]


def test_deny_term_is_caught_alongside_common_words():
    findings = scan_text("the findings mention SecretProject", ["secretproject"])
    assert [f.kind for f in findings] == ["deny_term"]


def test_scanner_own_test_file_is_exempt(tmp_path):
    (tmp_path / "tests").mkdir()
    target = tmp_path / "tests" / "test_check_no_secrets.py"
    target.write_text("/home/someone/x", encoding="utf-8")
    assert scan_repo(tmp_path, []) == []
