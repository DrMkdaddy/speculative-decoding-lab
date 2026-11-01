import unittest

from specdec.metrics import (
    accepted_by_position,
    diff,
    parse_metrics,
    request_spec_metrics,
)


METRICS = """\
# HELP llamacpp:tokens_predicted_total Number of generation tokens
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 1000
llamacpp:spec_decode_num_draft_tokens_total 80
llamacpp:spec_decode_num_accepted_tokens_total 56
llamacpp:spec_decode_num_drafts_total 20
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 18
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 14
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 10
"""


class MetricsTests(unittest.TestCase):
    def test_parse(self):
        metrics = parse_metrics(METRICS)
        self.assertEqual(metrics["llamacpp:tokens_predicted_total"], 1000)
        self.assertEqual(
            metrics['llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"}'],
            14,
        )

    def test_positions(self):
        positions = accepted_by_position(parse_metrics(METRICS))
        self.assertEqual(positions, {0: 18, 1: 14, 2: 10})

    def test_diff_clamps_reset(self):
        before = {"a": 10, "b": 5}
        after = {"a": 12, "b": 3}
        self.assertEqual(diff(before, after), {"a": 2, "b": 0})

    def test_request_metrics(self):
        before = parse_metrics(METRICS)
        after = parse_metrics(
            METRICS.replace("80", "160")
            .replace("56", "112")
            .replace("20", "40")
            .replace("18", "36")
            .replace('position="1"} 14', 'position="1"} 28')
            .replace('position="2"} 10', 'position="2"} 20')
        )
        result = request_spec_metrics(before, after)
        self.assertEqual(result["drafts"], 20)
        self.assertEqual(result["draft_tokens"], 80)
        self.assertEqual(result["accepted_tokens"], 56)
        self.assertAlmostEqual(result["acceptance"], 0.7)
        self.assertAlmostEqual(result["mean_accepted_length"], 3.8)
        self.assertAlmostEqual(result["acceptance_by_position"]["0"], 0.9)


if __name__ == "__main__":
    unittest.main()
