import unittest

from specdec.analyze import (
    acceptance_by_position,
    break_even,
    linear_fit,
    render_markdown,
    router_replay,
    summarize_config,
)
from specdec.router import AcceptanceRouter, classify_prompt


def row(config, speculative, prompt_id, cls, run, tps, acceptance=None, predicted_ms=100):
    drafts = 10 if speculative else 0
    accepted = int(round(acceptance * drafts)) if acceptance is not None else 0
    return {
        "config": config,
        "speculative": speculative,
        "n_max": 4 if speculative else None,
        "p_min": None,
        "prompt_id": prompt_id,
        "prompt_class": cls,
        "prompt": prompt_id,
        "run": run,
        "text": "same",
        "timings": {
            "predicted_per_second": tps,
            "predicted_ms": predicted_ms,
            "first_token_ms": 10,
        },
        "spec": {
            "drafts": drafts,
            "draft_tokens": accepted,
            "accepted_tokens": accepted,
            "acceptance": acceptance,
            "accepted_by_position": {"0": accepted / 2, "1": accepted / 2},
        },
        "wall_ms": predicted_ms + 1,
        "matches_baseline": True,
    }


def artifact():
    rows = []
    for run in range(2):
        rows += [
            row("baseline", False, "p1", "prose", run, 100, predicted_ms=100),
            row("n4", True, "p1", "prose", run, 120, acceptance=0.8, predicted_ms=80),
            row("baseline", False, "j1", "json", run, 100, predicted_ms=100),
            row("n4", True, "j1", "json", run, 90, acceptance=0.5, predicted_ms=110),
        ]
    return {
        "schema": "specdec/v1",
        "run_id": "test",
        "name": "pair",
        "hardware": {"gpu": "test"},
        "llama_cpp": {"version": "test", "commit": "abc"},
        "models": {"target": {"path": "/t"}, "draft": {"path": "/d"}},
        "run": {"max_tokens": 8, "runs": 2},
        "servers": {"baseline": {}, "n4": {}},
        "rows": rows,
        "validation": {"outputs_checked": 8, "output_mismatches": 0},
    }


class AnalysisTests(unittest.TestCase):
    def test_summary(self):
        summary = summarize_config(artifact(), "n4")
        self.assertAlmostEqual(summary["speedup_mean"], 1.05)
        self.assertAlmostEqual(summary["acceptance_mean"], 0.65)
        self.assertAlmostEqual(summary["mean_accepted_length"], 1.65)

    def test_positions(self):
        positions = acceptance_by_position(artifact(), "n4")
        self.assertAlmostEqual(positions[0], 0.325)

    def test_linear_fit(self):
        fit = linear_fit([0.5, 0.8], [0.9, 1.2])
        self.assertAlmostEqual(fit["slope"], 1.0)
        self.assertAlmostEqual(fit["acceptance_at_break_even"], 0.6)
        self.assertAlmostEqual(fit["r2"], 1.0)

    def test_break_even(self):
        fits = break_even(artifact())
        self.assertAlmostEqual(fits["n4"]["acceptance_at_break_even"], 0.6)

    def test_router_replay(self):
        replay = router_replay(artifact())
        self.assertEqual(replay["requests"], 4)
        self.assertGreater(replay["class_policy_ms"], 0)
        self.assertLessEqual(replay["oracle_ms"], replay["always_baseline_ms"])

    def test_render(self):
        text = render_markdown(artifact(), "test.json")
        self.assertIn("### Overall", text)
        self.assertIn("Router replay", text)
        self.assertIn("n4", text)


class RouterTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_prompt("Return a JSON array of 3 objects"), "json")
        self.assertEqual(classify_prompt("Write a Python function"), "code")
        self.assertEqual(classify_prompt("Solve the equation"), "math")
        self.assertEqual(classify_prompt("List ten ways"), "list")
        self.assertEqual(classify_prompt("Explain TCP"), "prose")

    def test_ewma(self):
        router = AcceptanceRouter(alpha=1.0, initial=1.1)
        prompt = "Explain TCP"
        self.assertEqual(router.choose(prompt), "spec")
        router.observe(prompt, 0.8)
        self.assertEqual(router.choose(prompt), "baseline")
        router.observe(prompt, 1.2)
        self.assertEqual(router.choose(prompt), "spec")


if __name__ == "__main__":
    unittest.main()
