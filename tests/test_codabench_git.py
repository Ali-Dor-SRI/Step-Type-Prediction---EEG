"""Exercise sync safety against temporary repositories; no network needed."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "codabench_git", Path(__file__).resolve().parents[1] / "scripts/codabench_git.py",
)
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


class GitSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin"
        self.run_git("init", "-b", "main", str(self.origin))
        self.run_git("-C", str(self.origin), "config", "user.name", "Sync test")
        self.run_git("-C", str(self.origin), "config", "user.email", "test@example.invalid")
        (self.origin / "codabench/analysis").mkdir(parents=True)
        (self.origin / "codabench/analysis/model.py").write_text("answer = 1\n")
        self.commit(self.origin, "initial")
        self.local = self.root / "local"
        self.run_git("clone", str(self.origin), str(self.local))
        self.run_git("-C", str(self.local), "config", "user.name", "Sync test")
        self.run_git("-C", str(self.local), "config", "user.email", "test@example.invalid")

    @staticmethod
    def run_git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()

    def commit(self, root, message):
        self.run_git("-C", str(root), "add", ".")
        self.run_git("-C", str(root), "commit", "-m", message)

    def test_fetch_leaves_working_files_unchanged_and_pull_fast_forwards(self):
        path = "codabench/analysis/model.py"
        (self.origin / path).write_text("answer = 2\n")
        self.commit(self.origin, "update")
        SYNC.fetch(self.local, "origin", "main")
        self.assertEqual((self.local / path).read_text(), "answer = 1\n")
        SYNC.pull(self.local, "origin", "main")
        self.assertEqual((self.local / path).read_text(), "answer = 2\n")

    def test_pull_preserves_local_edits(self):
        path = self.local / "codabench/analysis/model.py"
        path.write_text("answer = 'local edit'\n")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            SYNC.pull(self.local, "origin", "main")
        self.assertEqual(path.read_text(), "answer = 'local edit'\n")

    def test_pull_preserves_local_commits_when_remote_is_already_included(self):
        path = self.local / "codabench/analysis/model.py"
        path.write_text("answer = 'local commit'\n")
        self.commit(self.local, "local")
        old_head = SYNC.git(self.local, "rev-parse", "HEAD")
        SYNC.pull(self.local, "origin", "main")
        self.assertEqual(SYNC.git(self.local, "rev-parse", "HEAD"), old_head)

    def test_pull_preserves_divergent_commits(self):
        path = "codabench/analysis/model.py"
        (self.local / path).write_text("answer = 'local commit'\n")
        self.commit(self.local, "local")
        old_head = SYNC.git(self.local, "rev-parse", "HEAD")
        (self.origin / path).write_text("answer = 'remote commit'\n")
        self.commit(self.origin, "remote")
        with self.assertRaisesRegex(RuntimeError, "local commits"):
            SYNC.pull(self.local, "origin", "main")
        self.assertEqual(SYNC.git(self.local, "rev-parse", "HEAD"), old_head)
        self.assertEqual((self.local / path).read_text(), "answer = 'local commit'\n")

    def test_check_catches_untracked_code_and_tracked_weights(self):
        (self.local / "codabench/analysis/new.py").write_text("answer = 3\n")
        (self.local / "codabench/weights.pt").write_bytes(b"test artifact")
        self.run_git("-C", str(self.local), "add", "codabench/weights.pt")
        problems, parsed = SYNC.check_source(self.local)
        self.assertEqual(parsed, 1)
        self.assertTrue(any("Source is not tracked" in p for p in problems))
        self.assertTrue(any("Generated artifact tracked" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
