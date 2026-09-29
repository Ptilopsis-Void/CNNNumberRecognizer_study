import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_cnn.model import create_model
from mnist_cnn.training import _run_training_epoch, evaluate_model


class TrainingTests(unittest.TestCase):
    def test_one_synthetic_training_epoch_and_evaluation(self) -> None:
        torch.manual_seed(7)
        images = torch.rand(8, 1, 28, 28)
        labels = torch.arange(8) % 10
        data_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        model = create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        train_metrics = _run_training_epoch(
            model,
            data_loader,
            criterion,
            optimizer,
            device,
            epoch=1,
            total_epochs=1,
            log_interval=0,
        )
        test_metrics = evaluate_model(model, data_loader, criterion, device)

        self.assertGreater(train_metrics.loss, 0)
        self.assertGreater(test_metrics.loss, 0)
        self.assertGreaterEqual(test_metrics.accuracy, 0)
        self.assertLessEqual(test_metrics.accuracy, 100)


if __name__ == "__main__":
    unittest.main()

