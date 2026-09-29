import unittest

from mnist_cnn.config import DEFAULT_SETTINGS, PROJECT_ROOT


class SettingsTests(unittest.TestCase):
    def test_paths_are_anchored_to_project_root(self) -> None:
        self.assertEqual(DEFAULT_SETTINGS.project_root, PROJECT_ROOT)
        self.assertEqual(DEFAULT_SETTINGS.data_dir, PROJECT_ROOT / "data")
        self.assertEqual(
            DEFAULT_SETTINGS.model_path,
            PROJECT_ROOT / "artifacts" / "models" / "cnn_model.pth",
        )

    def test_overrides_do_not_mutate_defaults(self) -> None:
        changed = DEFAULT_SETTINGS.with_overrides(epochs=2)
        self.assertEqual(changed.epochs, 2)
        self.assertEqual(DEFAULT_SETTINGS.epochs, 10)


if __name__ == "__main__":
    unittest.main()

