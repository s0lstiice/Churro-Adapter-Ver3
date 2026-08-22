from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from universal_progress_monitor.progress_client import ProgressTask
from universal_progress_monitor.progress_dashboard import ProgressMonitor
from universal_progress_monitor.run_tracked import parse_progress


class ProgressParserTests(unittest.TestCase):
    def test_epoch_ratio(self) -> None:
        current, total, metrics = parse_progress("Epoch 3/8 loss=0.4")
        self.assertEqual((current, total), (3.0, 8.0))
        self.assertEqual(metrics, {})

    def test_json_metrics(self) -> None:
        current, total, metrics = parse_progress(
            json.dumps({"epoch": 2, "epochs": 6, "validation_iou": 0.85})
        )
        self.assertEqual((current, total), (2.0, 6.0))
        self.assertEqual(metrics["validation_iou"], 0.85)

    def test_percentage(self) -> None:
        self.assertEqual(parse_progress("rendering 42% complete")[:2], (42.0, 100.0))


class ReporterTests(unittest.TestCase):
    def test_exact_reporter_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = ProgressTask("unit test", 10, state_dir=temporary, task_id="unit-test")
            task.update(4, message="four done", metrics={"loss": 0.5})
            state = json.loads((Path(temporary) / "unit-test.json").read_text())
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["percent"], 40.0)
            self.assertEqual(state["metrics"]["loss"], 0.5)
            task.finish()
            state = json.loads((Path(temporary) / "unit-test.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["percent"], 100.0)

    def test_monitor_reads_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "states"
            task = ProgressTask("visible task", 5, state_dir=state_dir, task_id="visible")
            task.update(1)
            monitor = ProgressMonitor([Path(temporary)], state_dir)
            tasks = monitor._explicit_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["name"], "visible task")
            self.assertEqual(tasks[0]["percent"], 20.0)


if __name__ == "__main__":
    unittest.main()

