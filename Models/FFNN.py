import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import sklearn.model_selection

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser

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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1:]
        y_hat = self._nn(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer


seed = 10
sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=10000, rows_to_sample=1000, seed=seed)
sll_data_parser.load()

train, test = sklearn.model_selection.train_test_split(sll_data_parser.get_dataset(), test_size=0.3)
train_prepay = train.pop('zero_balance_code')
train.insert(len(train.columns), 'zero_balance_code', train_prepay)
test_prepay = test.pop('zero_balance_code')
test.insert(len(test.columns), 'zero_balance_code', test_prepay)

train_tensor = torch.tensor(train.values)
test_tensor = torch.tensor(test.values)

train_dataloader = DataLoader(train_tensor, batch_size=32)
test_dataloader = DataLoader(test_tensor, batch_size=32)

trainer = pl.Trainer()
nn_model = FFNN(len(train.columns) - 1).double()
trainer.fit(nn_model, train_dataloader, test_dataloader)


