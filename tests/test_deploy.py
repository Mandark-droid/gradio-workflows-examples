# tests/test_deploy.py
import subprocess

from pathlib import Path
from scripts.check_no_secrets import Finding
from scripts.deploy import build_file_list, EXAMPLES, SKIP_DIRS


def test_known_examples_are_registered():
    assert "shlokartha-voice" in EXAMPLES


def test_file_list_excludes_tests_and_caches(tmp_path):
    (tmp_path / "app.py").write_text("x", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"x")
    names = {p.name for p in build_file_list(tmp_path)}
    assert names == {"app.py"}


def test_file_list_keeps_workflow_and_config(tmp_path):
    for name in ("app.py", "workflow.json", "config.yaml", "requirements.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    names = {p.name for p in build_file_list(tmp_path)}
    assert names == {"app.py", "workflow.json", "config.yaml", "requirements.txt"}


def test_skip_dirs_covers_fixtures():
    assert "fixtures" in SKIP_DIRS


def test_file_list_excludes_nested_eval_results(tmp_path):
    # build_file_list checks path *parts* against SKIP_DIRS, unlike the old
    # ignore_patterns=["results/*", ...] passed to upload_folder, which is
    # fnmatched against the full relative path and never matches a nested
    # "eval/results/x.json" — that file was previewed as excluded but
    # uploaded anyway. This asserts the preview function itself is correct;
    # deploy() now derives the upload allow-list from this same function.
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "metrics.py").write_text("x", encoding="utf-8")
    (tmp_path / "eval" / "results").mkdir()
    (tmp_path / "eval" / "results" / "x.json").write_text("{}", encoding="utf-8")
    rel = {str(p.relative_to(tmp_path)).replace("\\", "/") for p in build_file_list(tmp_path)}
    assert "eval/metrics.py" in rel
    assert "eval/results/x.json" not in rel


def test_deploy_requires_a_token(monkeypatch):
    from scripts import deploy

    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Deterministic regardless of what this machine has stored: force the
    # "no credential anywhere" case rather than relying on the environment.
    monkeypatch.setattr(deploy, "_resolve_token", lambda: None)
    # deploy() scans the whole repo before it even looks at the token, so a
    # stray untracked file elsewhere in the working tree can make this test
    # fail for a reason that has nothing to do with token handling. Stub the
    # scanner so this test exercises only what it claims to.
    monkeypatch.setattr(deploy, "scan_repo", lambda *a, **k: [])
    assert deploy.deploy("shlokartha-voice", "owner/name", dry_run=False) == 2


def test_resolve_token_prefers_env_var(monkeypatch):
    from scripts import deploy

    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert deploy._resolve_token() == "env-token"


def test_resolve_token_falls_back_to_stored_credential(monkeypatch):
    from scripts import deploy

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "stored-cred")
    assert deploy._resolve_token() == "stored-cred"


def test_dry_run_skips_token_and_dirty_tree_checks(monkeypatch):
    from scripts import deploy

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(deploy, "_resolve_token", lambda: None)
    # Same isolation as test_deploy_requires_a_token above: this test is
    # about dry-run short-circuiting, not the scanner's view of whatever is
    # currently sitting untracked in the working tree.
    monkeypatch.setattr(deploy, "scan_repo", lambda *a, **k: [])

    real_run = subprocess.run

    def _guard(cmd, *args, **kwargs):
        # subprocess.run is shared with the secret scanner's own `git
        # ls-files` call, so only the dirty-tree check's `git status` may
        # not appear here — everything else passes through untouched.
        if cmd[:2] == ["git", "status"]:
            raise AssertionError("git status must not run in dry-run mode")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(deploy.subprocess, "run", _guard)
    assert deploy.deploy("shlokartha-voice", "owner/name", dry_run=True) == 0


def test_deploy_refuses_when_the_scanner_reports_a_finding(monkeypatch):
    from scripts import deploy

    # The mirror image of the two isolated tests above: with a real (faked)
    # finding, deploy() must refuse before it even looks at the token or the
    # working tree, so this behaviour stays covered now that those two stub
    # scan_repo out.
    finding = Finding(line_no=1, kind="token", snippet="a token-shaped string was here")
    fake_path = deploy.ROOT / "some" / "file.py"
    monkeypatch.setattr(deploy, "scan_repo", lambda *a, **k: [(fake_path, finding)])
    assert deploy.deploy("shlokartha-voice", "owner/name", dry_run=False) == 1


def test_deploy_refuses_dirty_tree(monkeypatch):
    from scripts import deploy

    monkeypatch.setattr(deploy, "_resolve_token", lambda: "fake-token")
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=a, returncode=0, stdout=" M workflow.json\n", stderr=""
        ),
    )
    assert deploy.deploy("shlokartha-voice", "owner/name", dry_run=False) == 1
