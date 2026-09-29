import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from mnist_cnn.config import DEFAULT_SETTINGS
from mnist_cnn.model import create_model
from mnist_cnn.prediction import load_model, prepare_image


class PredictionTests(unittest.TestCase):
    def test_preprocessing_centers_a_dark_digit_on_light_background(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "digit.png"
            image = Image.new("L", (80, 100), color=255)
            drawing = ImageDraw.Draw(image)
            drawing.line((40, 15, 40, 85), fill=0, width=12)
            image.save(image_path)

            tensor, processed = prepare_image(image_path, invert="auto")

        self.assertEqual(tuple(tensor.shape), (1, 1, 28, 28))
        self.assertEqual(processed.size, (28, 28))
        self.assertGreater(processed.getbbox()[3] - processed.getbbox()[1], 10)

    def test_raw_state_dict_checkpoint_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pth"
            torch.save(create_model().state_dict(), model_path)
            settings = DEFAULT_SETTINGS.with_overrides(model_path=model_path, device="cpu")
            loaded_model, device = load_model(settings)

        output = loaded_model(torch.zeros(2, 1, 28, 28, device=device))
        self.assertEqual(tuple(output.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()

