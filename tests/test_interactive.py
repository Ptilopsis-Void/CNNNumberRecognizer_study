import unittest
from pathlib import Path
from unittest.mock import patch

from mnist_cnn.interactive import _ask_use_existing_model, _clean_dragged_path


class InteractiveTests(unittest.TestCase):
    def test_cleans_double_quoted_dragged_path(self) -> None:
        path = _clean_dragged_path('  "C:\\Users\\student\\digit 7.png"  ')
        self.assertEqual(path, Path("C:/Users/student/digit 7.png"))

    def test_cleans_powershell_call_prefix(self) -> None:
        path = _clean_dragged_path("& 'C:\\Users\\student\\digit.png'")
        self.assertEqual(path, Path("C:/Users/student/digit.png"))

    @patch("builtins.input", return_value="")
    def test_existing_model_defaults_to_reuse(self, _: object) -> None:
        self.assertTrue(_ask_use_existing_model(Path("model.pth")))

    @patch("builtins.input", return_value="n")
    def test_existing_model_can_be_retrained(self, _: object) -> None:
        self.assertFalse(_ask_use_existing_model(Path("model.pth")))
