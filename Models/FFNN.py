import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning.metrics.functional as FM
import sklearn.model_selection

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser


def calculate_acc(y_hat, y):
    y_hat = torch.round(y_hat).type(torch.int)
    y = y.type(torch.int)
    val_acc = (y_hat == y).sum() / (y_hat.shape[0] * 1.0)
    return val_acc

def calculate_avg_acc_loss(outputs):
    avg_acc = 0.0
    avg_loss = 0.0
    total_count = 0.0
    for output in outputs:
        loss, y_hat, y = output
        acc = calculate_acc(y_hat, y)

        avg_loss += loss
        avg_acc += acc
        total_count += y_hat.shape[0]
    avg_loss /= total_count
    avg_acc /= total_count
    return avg_loss, avg_acc


class FFNN(pl.LightningModule):
    def __init__(self, input_len):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_len, 128),
                                 nn.Sigmoid(),
                                 nn.Linear(128, 256),
                                 nn.Sigmoid(),
                                 nn.Linear(256, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return self._nn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1:]
        y_hat = self._nn(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', calculate_acc(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1:]
        y_hat = self._nn(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', calculate_acc(y_hat, y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer


seed = 10
sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=100000, rows_to_sample=10000, seed=seed)
sll_data_parser.load()

train, val = sklearn.model_selection.train_test_split(sll_data_parser.get_dataset(), test_size=0.3)
train_prepay = train.pop('zero_balance_code')
train.insert(len(train.columns), 'zero_balance_code', train_prepay)
val_prepay = val.pop('zero_balance_code')
val.insert(len(val.columns), 'zero_balance_code', val_prepay)

train_tensor = torch.tensor(train.values)
val_tensor = torch.tensor(val.values)

train_dataloader = DataLoader(train_tensor, batch_size=32)
val_dataloader = DataLoader(val_tensor, batch_size=32)

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=5,
    mode='min',
    check_on_train_epoch_end=True
)
trainer = pl.Trainer(callbacks=[early_stopping])
nn_model = FFNN(len(train.columns) - 1).double()
trainer.fit(nn_model, train_dataloader, val_dataloader)

