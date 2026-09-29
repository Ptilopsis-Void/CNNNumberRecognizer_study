import unittest
from unittest.mock import patch

from mnist_cnn.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_train_arguments(self) -> None:
        arguments = build_parser().parse_args(["train", "--epochs", "2", "--device", "cpu"])
        self.assertEqual(arguments.command, "train")
        self.assertEqual(arguments.epochs, 2)
        self.assertEqual(arguments.device, "cpu")

    @patch("mnist_cnn.interactive.run_interactive_workflow", return_value=0)
    def test_no_arguments_runs_interactive_workflow(self, workflow: object) -> None:
        self.assertEqual(main([]), 0)
        workflow.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
