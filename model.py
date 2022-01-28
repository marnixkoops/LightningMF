from argparse import ArgumentParser
from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import functional as F


class MatrixFactorization(pl.LightningModule):
    """Torch Lightning implementation of the paper:
    Matrix Factorization Techniques for Recommender Systems
    by Koren, Y., Bell, R., & Volinsky, C. (2009).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 16,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super(MatrixFactorization, self).__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.user_embedding = torch.nn.Embedding(n_users, n_factors)
        self.item_embedding = torch.nn.Embedding(n_items, n_factors)
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        dot_product = torch.mul(user_vector, item_vector).sum(dim=1)

        rating = (
            dot_product + self.user_bias(user).view(-1) + self.item_bias(item).view(-1)
        )
        return rating

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        user, item, rating = batch
        pred = self(user, item)
        loss = F.mse_loss(pred, rating)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        user, item, rating = batch
        pred = self(user, item)
        val_loss = F.mse_loss(pred, rating)
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
        )
        return val_loss

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        user, item, rating = batch
        pred = self(user, item)
        test_loss = F.mse_loss(pred, rating)
        return test_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_users", type=int, default=None)
        parser.add_argument("--n_items", type=int, default=None)
        parser.add_argument("--n_factors", type=int, default=32)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=16)
        return parser


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(
        self, users: np.ndarray, items: np.ndarray, labels: np.ndarray
    ) -> None:
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> None:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.users[idx],
            self.items[idx],
            self.labels[idx],
        )


def main() -> None:
    pl.seed_everything(2022)
    parser = ArgumentParser()
    parser = MatrixFactorization.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # generate some random user-item interaction data for training
    n_users = 25
    n_items = 100
    n_interactions = 5000

    users = np.random.randint(0, n_users, size=n_interactions)
    items = np.random.randint(0, n_items, size=n_interactions)
    interactions = np.random.randint(0, 1, size=n_interactions).astype(float)

    val_ratio = 0.2
    dataset = InteractionDataset(users=users, items=items, labels=interactions)

    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)]
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MatrixFactorization(
        n_users=n_users, n_items=n_items, n_factors=args.n_factors
    )

    trainer = pl.Trainer.from_argparse_args(args, max_epochs=10)
    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
