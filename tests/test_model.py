import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        import torch

        from mnist_cnn.model import create_model

        output = create_model()(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (4, 10))


if __name__ == "__main__":
    unittest.main()

