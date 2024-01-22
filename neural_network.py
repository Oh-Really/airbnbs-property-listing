# %%
from typing import Any
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import os
import yaml


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv(Path(os.getcwd(), 'airbnb-property-listings', 'tabular_data', 'clean_tabular_data.csv'))

    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = torch.tensor(example.iloc[[7, 8, 9, 11, 12, 13, 14, 15, 16, 19]])
        label = torch.tensor(example['Price_Night'])
        return (features, label)

    def __len__(self):
        return len(self.data)
    

dataset = AirbnbNightlyPriceImageDataset()


train_set, test_set = random_split(dataset, [round(len(dataset)*0.9), round(len(dataset)*0.1)])
train_set, validation_set = random_split(train_set, [round(len(train_set)*0.8), round(len(train_set)*0.2)])
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)



# example = next(iter(train_loader))
# features, labels = example
# features = features.to(torch.float32)

class LinearRegression(torch.nn.Module):
    def __init__(self, nn_config):
        super().__init__()
        hidden_layers = nn_config['hidden_layer_width']
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10,hidden_layers[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[0],hidden_layers[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[1],1)
        )

    def forward(self, features):
        '''Performs forward pass on data. Executed when model is called.'''
        return self.layer(features)


# model = LinearRegression()
# print(model(features))
    
def get_nn_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config



def train(model, data_loader, config,  epochs=10):
    optimiser_method = eval(config['optimiser'])
    optimiser = optimiser_method(model.parameters(), lr = config['learning_rate'])
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            features = features.to(torch.float32)
            labels = labels.to(torch.float32).reshape(-1,1)
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(loss.item())
            # optimisation step
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1


def plot_validation_and_test_accuracy(model, data_loader_list: list):
    for loader in data_loader_list:
        writer = SummaryWriter()
        batch_idx = 0
        for batch in loader:
            features, labels = batch
            features = features.to(torch.float32)
            labels = labels.to(torch.float32).reshape(-1,1)
            prediction = model(features)
            print(prediction)

            writer.add_scalar('Prediction', prediction, batch_idx)
            batch_idx += 1

        writer - SummaryWriter()
        batch_idx = 0
        for batch in loader:
            features, labels = batch
            features = features.to(torch.float32)
            labels = labels.to(torch.float32).reshape(-1,1)
            writer.add_scalar('Predictions/Real_Value', labels, batch_idx)
            batch_idx += 1

    return



if __name__ == '__main__':
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)  
    model = LinearRegression(get_nn_config('nn_config.yaml'))
    #train(model, train_loader)
    plot_validation_and_test_accuracy(model, [train_loader, validation_loader])

# %%
