import unittest

from scripts.create_yolo26n_tuning_plan import build_plan


class TestYolo26nTuningPlan(unittest.TestCase):
    def test_optimizer_grid_has_sixty_jobs(self):
        config = {
            "cv": {"folds": 5},
            "tuning": {
                "learning_rates": [0.001, 0.005, 0.01],
                "batch_sizes": [8, 16],
                "optimizers": ["SGD", "AdamW"],
                "patience": 50,
            },
        }

        rows = build_plan(config)

        self.assertEqual(len(rows), 60)
        optimizers = set(row["optimizer"] for row in rows)
        self.assertEqual(optimizers, {"SGD", "AdamW"})
        self.assertEqual(set(row["patience"] for row in rows), {50})


if __name__ == "__main__":
    unittest.main()
