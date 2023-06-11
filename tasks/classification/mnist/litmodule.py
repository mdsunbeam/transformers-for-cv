from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy 

class MNISTLitModule(LightningModule):
    """LightningModule for MNIST classification.
    
    Six Main Tasks:
    
    - Initialization
    - Training Loop
    - Validation Loop
    - Test Loop
    - Prediction Loop
    - Optimizers and LR scheduler
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        self.net = net

        self.criterion = torch.nn.NLLLoss()

        # metrics across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # average loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # track best accuracy
        self.best_val_acc = MaxMetric()

        def forward(self, x: torch.Tensor):
            return self.net(x)
        
        def on_train_start(self):
            # lightning sanity checks validation step before training starts
            # reset any metrics stored from the initial checks
            self.val_acc.reset()
            self.val_loss.reset()
            self.best_val_acc.reset()

        def model_step(self, batch: Any):
            x, y = batch
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y

        def training_step(self, batch: Any, batch_idx: int):
            loss, preds, targets = self.model_step(batch)

            # update metrics
            self.train_loss(loss)
            self.train_acc(preds, targets)
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

            return loss
        
        def on_train_epoch_end(self):
            pass

        def validation_step(self, batch: Any, batch_idx: int):
            loss, preds, targets = self.model_step(batch)

            # update metrics
            self.val_loss(loss)
            self.val_acc(preds, targets)
            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        def on_validation_epoch_end(self):
            acc = self.val_acc.compute()
            self.best_val_acc(acc)
            self.log("val/best_acc", self.best_val_acc.compute(), sync_dist=True, prog_bar=True)

        def test_step(self, batch: Any, batch_idx: int):
            loss, preds, targets = self.model_step(batch)

            # update metrics
            self.test_loss(loss)
            self.test_acc(preds, targets)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        def on_test_epoch_end(self):
            pass

        def configure_optimizers(self):
            optimizer = self.optimizer(params=self.parameters())
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}
        
if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)



