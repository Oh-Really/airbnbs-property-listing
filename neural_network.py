# %%
from datetime import datetime
from time import time, strftime
from typing import Any
from torch.utils.data import Dataset, DataLoader, random_split
from torcheval.metrics.functional import r2_score
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import yaml
import json


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


train_set, test_set = random_split(dataset, [round(len(dataset)*0.894), round(len(dataset)*0.106)])
train_set, validation_set = random_split(train_set, [round(len(train_set)*0.8), round(len(train_set)*0.2)])

train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2)
validation_loader = DataLoader(validation_set, batch_size=2)
#data_loader_list = (train_loader, validation_loader, test_loader)


def get_nn_config(config_file):
    '''Return config file as Dict.'''
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# example = next(iter(train_loader))
# features, labels = example
# features = features.to(torch.float32)

class LinearRegression(torch.nn.Module):
    def __init__(self, nn_config):
        super().__init__()
        self.layer = torch.nn.Linear(10,1)
        hidden_layers = nn_config['hidden_layer_width']

        # self.layer = torch.nn.Sequential(
        #     torch.nn.Linear(10,hidden_layers[0]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_layers[0],hidden_layers[1]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_layers[1],1)
        # )
        # drop out
        # batch norm

    def forward(self, features):
        '''Performs forward pass on data. Executed when model is called.'''
        return self.layer(features)



def train_loop(model, train_loader, config=None,  epochs=10):
    start = time()
    # optimiser_method = eval(config['optimiser'])
    # optimiser = optimiser_method(model.parameters(), lr = config['learning_rate'])
    # TODO adam optimiser - try it out
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter()
    batch_idx = 0

    
    train_loss_avg = {} # lists of average losses over all epochs
    train_r2_avg = {}
    inference_time_avg = []
    for epoch in range(epochs):        
        model.train()
        train_loss_sum = []
        train_r2_sum = []
        inference_time = 0
        for batch in train_loader:
            features, labels = batch
            features = features.to(torch.float32)
            labels = labels.to(torch.float32).reshape(-1,1)
            start_prediction_time = time()
            prediction = model(features)
            end_prediction_time = time() - start_prediction_time
            end_prediction_time = end_prediction_time/len(labels)  # Average time taken for one prediction per batch
            inference_time += end_prediction_time 
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            train_loss_sum.append(loss.item())
            r2_value = r2_score(prediction, labels)
            train_r2_sum.append(r2_value)
            # if batch_idx == 0:
            #     print(loss)
            #     print(f'printing item: {loss.item()}')
            #     break
            # print(loss.item())
            # optimisation step
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('training_loss', loss.item(), batch_idx)
            print(f"batch number is {batch_idx}, loss is {loss.item()}")
            batch_idx += 1

        
        train_loss_avg[epoch] = train_loss_sum[-1]  # Add the last loss value of each epoch to the dict
        train_r2_avg[epoch] = train_r2_sum[-1]      # Add the last r2 value of each epoch to the dict
        inference_time_avg.append(inference_time)   # Add the sum of the average of all precition times for one epoch


    training_duration = np.round((time() - start), 2) # time in s
    train_metrics = {'RMSE:': list(train_loss_avg.values())[-1], "R2:" : list(train_r2_avg.values())[-1].item(), "train inference time:": inference_time_avg[-1]}
        

    generate_time_folders()


    return model, training_duration, train_metrics

    
def generate_time_folders():
    year, month, day = datetime.today().year, datetime.today().month, datetime.today().day
    hour, minute, second = strftime('%H'), strftime('%M'), strftime('%S')
    if len(str(day)) == 1:
        day = "0" + str(day)
    folder = f"{year}-{month}-{day}_{hour}-{minute}-{second}"
    os.chdir('models/neural_networks/regression/')
    os.mkdir(folder)
    os.chdir(folder)
    #save_model(model, config)
    #os.chdir('../../../../')


def eval_model(model, data_loader_list):
        '''
        Evaluates the trained model on the validation and test data sets.

        Parameters
        -----------
        model: The trained model returned by the train_loop function
        data_loader: A list of both the validation and test data loaders to be iterated over

        Returns
        -------
        val_metrics: A dictionary contraining the loss, r2 score, and average inference time of the validation set
        test_metrics: A dictionary contraining the loss, r2 score, and average inference time of the test set
        '''
        # val_metrics = {}
        # test_metrics = {}
        metrics = {}
        i = 0
        for loader in data_loader_list:
            loss_list = []
            r2_list = []   
            inference_times = []           
            model.eval()
            batch_idx = 0 # for debugging
            with torch.no_grad():
                for batch in loader:
                    features, labels = batch
                    features = features.to(torch.float32)
                    labels = labels.to(torch.float32).reshape(-1,1)
                    start_time = time()
                    prediction = model(features)
                    prediction_time = time() - start_time
                    inference_times.append(start_time - prediction_time)
                    loss = F.mse_loss(prediction, labels)
                    loss_list.append(loss.item())
                    r2_list.append(r2_score(prediction, labels))
                    # try:
                    #     r2_list.append(r2_score(prediction, labels))
                    # except:
                    #     print(f"i = {i}")
                    #     print(batch_idx)
                    #     print(features.shape)
                    #     print(labels.shape)
                    #     print(features)
                    #     print(labels)
                    # batch_idx += 1
                    #writer.add_scalar('validation_loss', loss.item(), batch_idx)
            
            # Extract loss and r2 score for the validation loader
            if i == 0:
                val_loss = loss_list[-1]
                val_r2 = r2_list[-1]
                val_inference_time = sum(inference_times)/len(loader)
                metrics['validation_set'] = {'RMSE:': val_loss, "R2:": val_r2.item(), "validation inference time:": val_inference_time}
                i+=1
            else:
                test_loss = loss_list[-1]
                test_r2 = r2_list[-1]
                test_inference_time = sum(inference_times)/len(loader)
                metrics['test_set'] = {'RMSE:': test_loss, "R2:": test_r2.item(), "test inference time:": test_inference_time}
        
        # # Extract the test loss and r2
        # test_loss = loss_list[-1]
        # test_r2 = r2_list[-1]
        # test_inference_time = sum(inference_times)/len(loader)
        # metrics = {'test_set': {'test_loss:': {test_loss}, "test_r2:" :{test_r2}, "test inference time:": {test_inference_time}}}

        return metrics

# Debugging main statement
# if __name__ == "__main__":
#     config = get_nn_config('nn_config.yaml')
#     model = LinearRegression(config)
#     eval_model(model, [validation_loader, test_loader])




def save_model(model, config, train_metrics, training_duration):
    if issubclass(model.__class__, torch.nn.Module):
        torch.save(model.state_dict(), 'model.pt')
        hyperparams = json.dumps(config)
        with open("hyperparameters.json", 'w+') as hyperparam_file:
            hyperparam_file.write(hyperparams)
        # Get metrics, then dump into file called metrics.json
        metrics = eval_model(model, [validation_loader, test_loader])
        metrics['training_set'] = train_metrics
        metrics['training_duration'] = training_duration
        print(type(metrics))
        metrics = json.dumps(metrics)
        with open("metrics.json", 'w+') as metrics_file:
            metrics_file.write(metrics)
        os.chdir('../../../../')
        


if __name__ == "__main__":
    config = get_nn_config('nn_config.yaml')
    model = LinearRegression(config)
    model, training_duration, train_metrics = train_loop(model, train_loader, config)
    save_model(model, config, train_metrics, training_duration)
# %%
