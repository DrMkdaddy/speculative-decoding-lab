import json
import tempfile
import unittest
from pathlib import Path

from specdec.config import ConfigError, load_config


MANIFEST = {
    "name": "pair",
    "target": {"filename": "target.gguf"},
    "draft": {"filename": "draft.gguf"},
    "server": {"context": 2048, "port": 8091},
    "run": {"max_tokens": 32, "runs": 2},
    "configs": [
        {"name": "baseline", "spec": None},
        {"name": "n4", "spec": {"n_max": 4}},
        {"name": "n6-p0.6", "spec": {"n_max": 6, "p_min": 0.6}},
    ],
}


class ConfigTests(unittest.TestCase):
    def _write(self, data):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "config.json"
        path.write_text(json.dumps(data))
        self.addCleanup(tmp.cleanup)
        return path

    def test_paths(self):
        cfg = load_config(self._write(MANIFEST), model_dir="/models")
        self.assertEqual(cfg.target.path, Path("/models/target.gguf"))
        self.assertEqual(cfg.draft.path, Path("/models/draft.gguf"))
        self.assertEqual(cfg.sweep("n6-p0.6").p_min, 0.6)
        self.assertFalse(cfg.sweep("baseline").speculation)

    def test_requires_baseline(self):
        data = dict(MANIFEST)
        data["configs"] = [{"name": "n4", "spec": {"n_max": 4}}]
        with self.assertRaises(ConfigError):
            load_config(self._write(data), model_dir="/models")

    def test_duplicate_names(self):
        data = dict(MANIFEST)
        data["configs"] = [
            {"name": "baseline", "spec": None},
            {"name": "baseline", "spec": None},
        ]
        with self.assertRaises(ConfigError):
            load_config(self._write(data), model_dir="/models")


if __name__ == "__main__":
    unittest.main()
