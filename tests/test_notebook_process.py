"""Real child processes exercise notebook output, failures and cancellation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from notebook_process import run_logged


class NotebookProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log = Path(temporary.name) / "console.log"
        self.children = []
        real_popen = subprocess.Popen

        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            self.children.append(child)
            return child

        mock = patch("notebook_process.subprocess.Popen", side_effect=spawn)
        mock.start()
        self.addCleanup(mock.stop)
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
                child.wait()
            if child.stdout:
                child.stdout.close()

    def run_child(self, source, **kwargs):
        return run_logged([sys.executable, "-u", "-c", source], self.log, **kwargs)

    def test_streams_stdout_stderr_and_appends(self):
        self.log.write_text("previous run\n", encoding="utf-8")
        lines = []
        self.run_child("import sys; print('epoch 1'); print('warning', file=sys.stderr)", emit=lines.append)
        self.assertEqual(lines, ["epoch 1\n", "warning\n"])
        self.assertEqual(self.log.read_text(encoding="utf-8"), "previous run\nepoch 1\nwarning\n")

    def test_nonzero_exit_keeps_diagnostic(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.run_child("print('training failed'); raise SystemExit(7)", emit=lambda line: None)
        self.assertEqual(failure.exception.returncode, 7)
        self.assertIn("training failed", self.log.read_text(encoding="utf-8"))

    def assert_cancelled(self, exception, setup="", stop_timeout=1):
        def cancel(line):
            raise exception

        with self.assertRaises(type(exception)):
            self.run_child(
                setup + "import time; print('READY', flush=True); time.sleep(30)",
                emit=cancel, stop_timeout=stop_timeout,
            )
        self.assertIsNotNone(self.children[0].poll(), "training child survived cancellation")
        self.assertTrue(self.children[0].stdout.closed)
        self.assertEqual(self.log.read_text(encoding="utf-8"), "READY\n")

    def test_keyboard_interrupt_stops_training_child(self):
        self.assert_cancelled(KeyboardInterrupt())

    def test_output_failure_stops_training_child(self):
        self.assert_cancelled(OSError("notebook output unavailable"))

    @unittest.skipUnless(os.name == "posix", "SIGTERM handling requires POSIX")
    def test_unresponsive_child_is_killed_after_timeout(self):
        self.assert_cancelled(
            KeyboardInterrupt(),
            setup="import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); ",
            stop_timeout=0.1,
        )


if __name__ == "__main__":
    unittest.main()
