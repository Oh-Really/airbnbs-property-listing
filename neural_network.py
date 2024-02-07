# %%
from datetime import datetime
from time import time, strftime
from typing import Any
from torch.utils.data import Dataset, DataLoader, random_split
from torcheval.metrics.functional import r2_score
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random
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
        #self.layer = torch.nn.Linear(10,1)
        hidden_layers = nn_config['hidden_layer_width']

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10,hidden_layers[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[0],hidden_layers[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[1],1)
        )
        # drop out
        # batch norm

    def forward(self, features):
        '''Performs forward pass on data. Executed when model is called.'''
        return self.layer(features)



def train_loop(model, train_loader, config=None,  epochs=10):
    '''
    Function to train and obtain training metrics for a given model.

    Paramaters
    ----------
    model: Model that inherits from the torch.nn.Module class
    train_loader: DataLoader object for the training set
    config: Config file of the hyperparamaters of the model
    epochs: Number of epochs to train the model over
    
    Returns
    -------
    model: Trained model with updated weights after training
    training_duration: Time taken to train the model in s
    train_metrics: Dictionary of training metrics, including RMSE, R2, and inference time
    '''
    start = time()
    optimiser_method = eval(config['optimiser'])
    optimiser = optimiser_method(model.parameters(), lr = config['learning_rate'])
    #  optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

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
            #print(f"batch number is {batch_idx}, loss is {loss.item()}")
            batch_idx += 1

        
        train_loss_avg[epoch] = train_loss_sum[-1]  # Add the last loss value of each epoch to the dict
        train_r2_avg[epoch] = train_r2_sum[-1]      # Add the last r2 value of each epoch to the dict
        inference_time_avg.append(inference_time)   # Add the sum of the average of all precition times for one epoch


    training_duration = np.round((time() - start), 2) # time in s
    train_metrics = {'RMSE:': list(train_loss_avg.values())[-1], "R2:" : list(train_r2_avg.values())[-1].item(), "train inference time:": inference_time_avg[-1]}
        

    #generate_time_folders()


    return model, training_duration, train_metrics

    
def generate_time_folders():
    '''
    Function to create folders with folder name of current time.

    '''
    year, month, day = datetime.today().year, datetime.today().month, datetime.today().day
    hour, minute, second = strftime('%H'), strftime('%M'), strftime('%S')
    if len(str(day)) == 1:
        day = "0" + str(day)
    folder = f"{year}-{month}-{day}_{hour}-{minute}-{second}"
    os.chdir('models/neural_networks/regression/')
    os.mkdir(folder)
    os.chdir(folder)
    #os.chdir('../../../../')


def eval_model(model, data_loader_list, train_metrics, training_duration):
        '''
        Evaluates the trained model on the validation and test data sets.

        Parameters
        -----------
        model: The trained model returned by the train_loop function
        data_loader: A list of both the validation and test data loaders to be iterated over

        Returns
        -------
        metrics: A dictionary contraining the loss, r2 score, and average inference time of the validation and test sets
        '''
        
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

        metrics['training_set'] = train_metrics
        metrics['training_duration'] = training_duration
        return metrics

# Debugging main statement
# if __name__ == "__main__":
#     config = get_nn_config('nn_config.yaml')
#     model = LinearRegression(config)
#     eval_model(model, [validation_loader, test_loader])




def save_model(model, config, metrics):
    ''''
    Saves the model, its hyperparameters, and results metrics.

    Parameters
    ---------
    model: The trained model returned by the train_loop function
    config: The config associated with the model
    train_metrics: A dictionary object containing the results from the training 
    training_duration: Float value for the length of time it took train_loop to train the model

    Returns
    -------
    metrics: Dictionary returned by eval_model, the dictionary of validation and test sets
    '''
    if issubclass(model.__class__, torch.nn.Module):
        torch.save(model.state_dict(), 'model.pt')
        hyperparams = json.dumps(config)
        with open("hyperparameters.json", 'w+') as hyperparam_file:
            hyperparam_file.write(hyperparams)
        ## Get metrics, then dump into file called metrics.json
        ## metrics = eval_model(model, [validation_loader, test_loader])
        ## metrics['training_set'] = train_metrics
        ## metrics['training_duration'] = training_duration
        metrics = json.dumps(metrics)
        with open("metrics.json", 'w+') as metrics_file:
            metrics_file.write(metrics)
        os.chdir('../../../../')

    return metrics
        

def generate_nn_configs():
    config_list = []
    for lr in [0.1, 0.001, 0.001]:
        for i in range(6):
            layer1 = random.randint(10, 15)
            layer2 = random.randint(2, 5)
            config = {}
            config['optimiser'] = 'torch.optim.Adam'
            config['learning_rate'] = lr
            config['hidden_layer_width'] = [layer1, layer2]
            config_list.append(config)
    
    return config_list


def find_best_nn():
    config_list = generate_nn_configs()
    best_rmse = np.inf
    for config in config_list:
        model = LinearRegression(config)
        model, training_duration, train_metrics = train_loop(model, train_loader, config)
        metrics = eval_model(model, [validation_loader, test_loader], train_metrics, training_duration)
        #metrics = save_model(model, config, train_metrics, training_duration)
        if metrics['test_set']['RMSE:'] < best_rmse:
            best_model = model
            best_metrics = metrics
            best_config = config
            best_rmse = metrics['test_set']['RMSE:']

    generate_time_folders()
    save_model(best_model, best_config, best_metrics)
    return best_model, best_metrics, best_config

if __name__ == "__main__":
    model, metrics, config = find_best_nn()
    # config = get_nn_config('nn_config.yaml')
    # model = LinearRegression(config)
    # model, training_duration, train_metrics = train_loop(model, train_loader, config)
    # save_model(model, config, train_metrics, training_duration)
# %%
