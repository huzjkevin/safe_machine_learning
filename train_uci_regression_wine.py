'''
 * User: Hojun Lim
 * Date: 2020-04-25
'''
import torch
import torchvision
from torch import optim
import os
from utils.train_utils import *

from utils.eval_utils import model_fn
import utils.train_utils as tu
import utils.dataset
import utils as u
import cv2 as cv
import numpy as np
from utils.dataset import WineDataset
from torch.utils.data import DataLoader
import pandas as pd
from model.Wine_FC import Wine_FC
import torch.nn as nn
from utils.utils import split_wine_dataset
from utils.eval_utils import compute_mean_and_variance, evaluate_with_NLL, evaluate_with_RMSE, compute_test_loss
from utils.utils import get_args_from_yaml, draw_loss_trend_figure

def train(train_set_dir, num_epochs, ckpt_save_interval, ckpt_dir, batch_size, learning_rate, input_dim, dropout_prop,
          num_worker, output_dir, test_set_dir):
    # train step
    # train_dataset = WineDataset(data_dir=train_set_dir, transform=None)
    train_dataset = WineDataset(train_set_dir, transform=None)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_worker)  # TODO current Environment is window, so I set num_worker 0 but in linux should be higher number.

    # prepare model
    model = Wine_FC(input_dim, prop=dropout_prop)
    model = model.float()
    model = model.to(device)

    # prepare training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define starting iteration/epochs.
    # Will use checkpoints in the future when running on clusters
    if ckpt is not None:
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=ckpt)
    elif os.path.isfile(os.path.join(ckpt_dir, "sigterm_ckpt.pth")):
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer,
                                                             filename=os.path.join(ckpt_dir, "sigterm_ckpt.pth"))
    else:
        starting_iteration, starting_epoch = 0, 0

    # variables for saving the loss values from training set and testing set per epoch
    train_loss_list = []
    test_loss_list = []

    ## Training step
    model.train()  # set model in train mode

    _it = 0  # variable for counting the interations
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0

        # train for one epoch
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()  # clear gradients from previous iteration

            input, label = batch

            # preprocessing inputs and labels in order to prevent some runtime error( i.e tensor dim or datatype not matching)
            input = input.to(device)
            input = input.float()
            label = label.to(device)
            label = label.float()
            label = label.view(-1, 1)  # Tensor reshape

            model_output = model(input)

            MSELoss = nn.MSELoss()  # set loss function as MSE

            loss = MSELoss(model_output, label)
            epoch_loss += (loss.item() * np.shape(input)[0])
            _it += 1

            loss.backward()  # compute gradients
            optimizer.step()  # update parameters

        # save trained model
        trained_epoch = epoch + 1
        print("Current Epoch: %d" % trained_epoch)
        print("Epoch loss: ", epoch_loss / len(train_dataset))

        # save model
        if trained_epoch % ckpt_save_interval == 0:
            print('Saving Checkpoint')
            ckpt_name = os.path.join(ckpt_dir, "ckpt_e%d" % trained_epoch)
            save_checkpoint(checkpoint_state(model, optimizer, trained_epoch, _it), filename=ckpt_name)

        ###
        train_loss_list.append(epoch_loss / len(train_dataset))


        test_loss = test(test_set_dir, batch_size, num_worker, dropout_prop, trained_epoch, ckpt_dir, input_dim)
        test_loss_list.append(test_loss)
        ###

    draw_loss_trend_figure('Wine dataset Loss Figure', train_loss_list, test_loss_list, num_epochs, output_dir)


def test(test_set_dir, batch_size, num_worker, dropout_prop, num_epochs, ckpt_dir, input_dim):
    # Testing the model
    print('Start testing')
    # test_dataset = WineDataset(data_dir=test_set_dir, transform=None)
    test_dataset = WineDataset(test_set_dir, transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    model = Wine_FC(input_dim, prop=dropout_prop)
    model = model.float()
    model = model.to(device)

    load_epoch = num_epochs  # load_epoch implies which model with given epoches to load from ckpt
    cur_ckpt = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e%d" % (load_epoch)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])



    testing_loss = 0
    epoch_test_loss = 0
    model.train()  # in order to turn on the dropout for MC dropout
    with torch.no_grad():
        for iter, batch in enumerate(test_dataloader):
            input, label = batch

            # preprocessing inputs and labels in order to prevent some runtime error( i.e tensor dim or datatype not matching)
            input = input.to(device)
            input = input.float()
            label = label.to(device)
            label = label.float()
            label = label.view(-1, 1)  # Tensor reshape

            model_output = model(input)

            MSELoss = nn.MSELoss()  # set loss function as MSE
            loss = MSELoss(model_output, label)


            testing_loss = (loss.item() * np.shape(input)[0])
            epoch_test_loss += testing_loss

    print("Test loss: ", epoch_test_loss / len(test_dataset))
    return epoch_test_loss / len(test_dataset)


def eval_with_MCDropout(test_set_dir, batch_size, num_worker, dropout_prop, num_epochs, ckpt_dir, num_networks, input_dim):
    print('Start Evaluating with MC Dropout')
    # test_dataset = WineDataset(data_dir=test_set_dir, transform=None)
    test_dataset = WineDataset(test_set_dir, transform=None)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    model = Wine_FC(input_dim, prop=dropout_prop)
    model = model.float()
    model = model.to(device)

    load_epoch = num_epochs  # load_epoch implies which model with given epoches to load from ckpt
    cur_ckpt = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e%d" % (load_epoch)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])

    model.train()

    NLL_list = None  # save the computed NLL loss
    RMSE_list = None

    with torch.no_grad():
        for batch in test_dataloader:

            input, label = batch

            # preprocessing inputs and labels in order to prevent some runtime error( i.e tensor dim or datatype not matching)
            input = input.to(device)
            input = input.float()
            label = label.to(device)
            label = label.float()
            label = label.view(-1, 1)  # Tensor reshape

            samples = None
            for i in range(
                    num_networks):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    samples = model(input).tolist()
                else:
                    model_output = model(input).tolist()
                    samples = np.append(samples, np.asarray(model_output), axis=1)

            # np.shape(samples) = [batch_size, num_networks]
            # print('model_output', model_output, 'samples', samples[-2] )

            # Compute the mean and variance by MC dropout approximation

            # print('sample',samples)
            mean, var = compute_mean_and_variance(samples, num_networks)

            NLL = evaluate_with_NLL(mean, var,
                                    label.tolist())  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, label.tolist())

            if NLL_list is None:
                NLL_list = NLL
                RMSE_list = RMSE
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))

        # print(NLL_list, np.shape(NLL_list))

        print(min(NLL_list), max(NLL_list))

        # export the evaluated NLL results to csv file
        import pandas as pd

        df = pd.DataFrame(NLL_list)

        df.to_csv('NLL_result.csv', index=False)

        print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
        print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))


if __name__ == '__main__':
    
    config =  get_args_from_yaml('./configs/all_config.yml')
    print(config['output_dir']['wine'])

    output_dir = config['output_dir']['wine']
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # some cfgs, some cfg will be used in the future

    wine_data_dir = config['data']['dir']['wine']['data_dir']
    train_set_dir = config['data']['dir']['wine']['train_set_dir']
    test_set_dir = config['data']['dir']['wine']['test_set_dir']
    ckpt = None

    dropout_prob = config['parameters']['dropout_prob']
    num_epochs = config['parameters']['num_epochs']
    ckpt_save_interval = config['ckpt']['ckpt_save_interval']
    batch_size = config['parameters']['batch_size']
    learning_rate = config['parameters']['learning_rate']
    input_dim = config['data']['input_dim']['wine']['input_dim']  # dataset has 11 attributes
    num_worker = 0
    num_networks = config['parameters']['num_networks']

     # split dataset into train and test dataset
    split_wine_dataset(wine_data_dir, train_set_dir, test_set_dir)

    ## run the following train function when you want to train the model and save it
    train(train_set_dir, num_epochs, ckpt_save_interval, ckpt_dir, batch_size, learning_rate, input_dim, dropout_prob,num_worker, output_dir, test_set_dir)

    ## run the following test function when you want to evaluate the test accurracy and test loss
    test(test_set_dir, batch_size, num_worker, dropout_prob, num_epochs, ckpt_dir, input_dim)

    ## given the saved model from train step, compute the mean and variance with NLL, RMSE
    eval_with_MCDropout(test_set_dir, batch_size, num_worker, dropout_prob, num_epochs, ckpt_dir, num_networks,input_dim)
