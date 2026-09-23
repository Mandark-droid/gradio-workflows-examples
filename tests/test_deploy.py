# tests/test_deploy.py
import subprocess

import pytest
from pathlib import Path
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


def test_deploy_requires_a_token(monkeypatch):
    from scripts import deploy

    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Deterministic regardless of what this machine has stored: force the
    # "no credential anywhere" case rather than relying on the environment.
    monkeypatch.setattr(deploy, "_resolve_token", lambda: None)
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
