import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

EPOCHS = 150
BATCH_SIZE = 5
LEARNING_RATE = 0.00005
device = torch.device("cuda:0")


class Data(Dataset):
    """My dataset"""

    def __init__(self, x_data, y_data):
        """
        """
        super(Dataset).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(2048, 1024)
        self.layer_2 = nn.Linear(1024, 64)
        self.layer_3 = nn.Linear(64, 128)
        self.layer_out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.relu(self.layer_out(x))

        return x


def binary_acc(y_pred, y_test):
    y_pred = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = 1 * (y_pred == y_test.int()).sum().item()
    acc = correct_results_sum / y_test.shape[0]

    return acc


def make_datasets(dataset):
    x, y = dataset[:, :2048], dataset[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)
    train_data = Data(x_train.float(),
                      y_train.float())
    test_data = Data(x_test.float(),
                     y_test.float())
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    return train_loader, test_loader, y_test


def train(dataset):
    model = binaryClassification()
    model.to(device)
    # print(model)
    train_loader, test_loader, y_test = make_datasets(dataset)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
    torch.save(model.state_dict(), "D:\\5th\Honours\Code\models\weights/cnn.pt")
    y_pred_list = []
    y_test = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_test_pred = model(x_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu())
            y_test.append(y_batch.cpu())
    y_pred_list = [item for sublist in y_pred_list for item in sublist]
    y_test = [item for sublist in y_test for item in sublist]
    timer = time.perf_counter()
    model(x_batch)
    timer2 = time.perf_counter()
    print(f"Finished apks in short time in {timer2 - timer:0.4f} seconds")
    print(classification_report(y_test, y_pred_list, digits=4))

def test(dataset, path):
    model = binaryClassification()
    model.to(device)
    model.load_state_dict(torch.load(path))
    train_loader, test_loader, y_test = make_datasets(dataset)
    y_pred_list = []
    y_test = []
    model.eval()
    batch_acc = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)

        acc = binary_acc(y_pred, y_batch)
        batch_acc += acc
        print(classification_report(y_pred, y_batch))