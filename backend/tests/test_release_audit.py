import subprocess

from scripts.audit_release import SELECTED_MODEL, audit_release_tree, audit_repository


def test_audit_rejects_excluded_and_extra_model_paths(tmp_path) -> None:
    result = audit_release_tree(
        root=tmp_path,
        tracked_files=(
            "README.md",
            "docs/private.md",
            "backend/app/evaluation/run.py",
            "backend/app/post_training/train.py",
            "models/other.gguf",
        ),
        lfs_files=("models/other.gguf",),
        blob_sizes={"README.md": 10, "docs/private.md": 10},
        staged_texts={},
    )
    assert result.ok is False
    assert "docs/private.md" in result.forbidden_paths
    assert "backend/app/evaluation/run.py" in result.forbidden_paths
    assert "backend/app/post_training/train.py" in result.forbidden_paths
    assert "models/other.gguf" in result.unexpected_models


def test_audit_accepts_minimal_release_fixture(tmp_path) -> None:
    result = audit_release_tree(
        root=tmp_path,
        tracked_files=("README.md", "main.py", SELECTED_MODEL),
        lfs_files=(SELECTED_MODEL,),
        blob_sizes={"README.md": 100, "main.py": 100},
        staged_texts={"README.md": "Paper-Agent"},
    )
    assert result.ok is True


def test_audit_rejects_large_git_blob_and_secret(tmp_path) -> None:
    result = audit_release_tree(
        root=tmp_path,
        tracked_files=("README.md", SELECTED_MODEL),
        lfs_files=(SELECTED_MODEL,),
        blob_sizes={"README.md": 101 * 1024 * 1024},
        staged_texts={"README.md": "AGENT_API_KEY=sk-private-real-value"},
    )
    assert result.ok is False
    assert result.large_blobs == ("README.md",)
    assert result.secret_findings == ("README.md",)


def test_audit_requires_selected_model_to_be_lfs(tmp_path) -> None:
    result = audit_release_tree(
        root=tmp_path,
        tracked_files=("README.md", SELECTED_MODEL),
        lfs_files=(),
        blob_sizes={"README.md": 100},
        staged_texts={},
    )
    assert result.ok is False
    assert result.missing_lfs == (SELECTED_MODEL,)


def test_audit_does_not_join_empty_key_with_following_line(tmp_path) -> None:
    result = audit_release_tree(
        root=tmp_path,
        tracked_files=(".env.example", SELECTED_MODEL),
        lfs_files=(SELECTED_MODEL,),
        blob_sizes={".env.example": 40},
        staged_texts={".env.example": "AGENT_API_KEY=\nAGENT_BASE_URL=https://example.invalid\n"},
    )
    assert result.secret_findings == ()


def test_repository_audit_reads_committed_content_not_dirty_worktree(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"], cwd=tmp_path, check=True
    )
    (tmp_path / "README.md").write_text("Paper-Agent\n", encoding="utf-8")
    model = tmp_path / SELECTED_MODEL
    model.parent.mkdir(parents=True)
    model.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'a' * 64}\nsize 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "README.md", SELECTED_MODEL], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text(
        "AGENT_API_KEY=sk-proj-" + "a" * 40 + "\n", encoding="utf-8"
    )

    assert audit_repository(tmp_path).ok is True
