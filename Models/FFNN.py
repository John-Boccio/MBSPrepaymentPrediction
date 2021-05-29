import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sklearn.model_selection
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import numpy

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser


def calculate_acc(y_hat, y):
    y_hat = torch.round(y_hat).type(torch.int)
    y = y.type(torch.int)
    val_acc = (y_hat == y).sum() / (y_hat.shape[0] * 1.0)
    return val_acc


class FFNN(pl.LightningModule):
    def __init__(self, input_len):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_len, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Sigmoid(),
                                 nn.Linear(256, 1))
        # Random weight initialization
        for layer in self._nn:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        return self._nn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1:]
        y_hat = self._nn(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=torch.tensor(10.0))
        self.log('train_loss', loss)
        self.log('train_acc', calculate_acc(F.sigmoid(y_hat), y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1:]
        y_hat = self._nn(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=torch.tensor(10.0))
        self.log('val_loss', loss)
        self.log('val_acc', calculate_acc(F.sigmoid(y_hat), y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1:]
        y_hat = self._nn(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=torch.tensor(10.0))
        self.log('test_loss', loss)
        self.log('test_acc', calculate_acc(F.sigmoid(y_hat), y))
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.91),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]


sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=250000, rows_to_sample=100000)
sll_data_parser.load()

df = sll_data_parser.get_dataset()
#scaler = preprocessing.StandardScaler(with_mean=False)
#df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)

train, val = sklearn.model_selection.train_test_split(df, test_size=0.2)
train, test = sklearn.model_selection.train_test_split(train, test_size=0.2)
# train 0.6, val 0.2, test 0.2
train_prepay = train.pop('zero_balance_code')
train.insert(len(train.columns), 'zero_balance_code', train_prepay)
val_prepay = val.pop('zero_balance_code')
val.insert(len(val.columns), 'zero_balance_code', val_prepay)
test_prepay = test.pop('zero_balance_code')
test.insert(len(test.columns), 'zero_balance_code', test_prepay)

train_tensor = torch.tensor(train.values)
val_tensor = torch.tensor(val.values)
test_tensor = torch.tensor(test.values)

train_dataloader = DataLoader(train_tensor, batch_size=64, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_tensor, batch_size=64, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_tensor, batch_size=64, num_workers=8, pin_memory=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=10,
    mode='min',
    check_on_train_epoch_end=True
)
trainer = pl.Trainer(min_epochs=20, max_epochs=150, callbacks=[early_stopping], gpus=-1)
nn_model = FFNN(len(test.columns)-1).double()

training = 0
if training:
    trainer.fit(nn_model, train_dataloader, val_dataloader)
    trainer.test(test_dataloaders=[test_dataloader])
else:
    nn_model = FFNN.load_from_checkpoint("/home/john/PycharmProjects/MBSPrepaymentPrediction/Models/lightning_logs/version_7/checkpoints/epoch=16-step=1055886.ckpt", input_len=len(test.columns)-1).double()
    conf_mat = torchmetrics.ConfusionMatrix(2)
    cm = torch.zeros(2, 2)
    for batch in test_dataloader:
        x, y = batch[:, :-1], batch[:, -1:].type(torch.int)
        y_hat = nn_model(x)
        cm += conf_mat(torch.clamp(y_hat, 0.0, 1.0), y)
    plt.figure()
    sns.heatmap(cm.type(torch.int), annot=True, cmap='Blues', fmt='d')
    plt.show()
