import socket
from datetime import datetime
import os
import torch


def get_run_name(title):
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H:%M') + '_' + socket.gethostname() + '_' + title


def average_model_parameters(parent_folder, train_model, pth_name='best_epoch.pth', save_path='average_best_epoch.pth'):
    # Get all subfolders of the model
    model_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    model_parameters = []

    # Load each model and retrieve its parameters
    for folder in model_folders:
        model_path = os.path.join(parent_folder, folder, pth_name)

        model = train_model
        model.load_state_dict(torch.load(model_path))
        model_parameters.append(model.state_dict())

    # Calculate the average value of parameters for all models
    average_state_dict = {}
    for key in model_parameters[0].keys():
        average_state_dict[key] = sum([params[key] for params in model_parameters]) / len(model_parameters)

    # Create a new model and load the average parameters
    average_model = train_model
    average_model.load_state_dict(average_state_dict)

    # Save average parameters to a file
    average_model_path = save_path
    torch.save(average_model.state_dict(), average_model_path)

    print(f'Average model parameters saved to {average_model_path}')
# socket.gethostname() get the current host name