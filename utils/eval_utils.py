import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils.dataset import *
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import *
from model import Wine_FC
# from utils.dataset import WineDataset


def model_fn(model, data):
    rtn_dict = {}
    #unpack data
    input, target = data
    # input, target, _ = data # code for normalization new normalization
    #Move data to GPU
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    pred = model(input)

    loss = model.loss_fn(pred, target)

    return loss

def model_fn_eval(model, eval_loader):
    eval_loss = 0.0
    for it, batch in enumerate(eval_loader):

        # loss = model_fn(model, batch)

        input, target = batch
        # input, target, stat = batch # code for normalization new normalization
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
        pred = model(input)

        #[[X_mean, Y_mean],
        # [X_sigma, Y_sigma]]
        # pred = stat[1, 1] * pred + stat[0, 1] # code for normalization new normalization
        # target = stat[1, 1] * target + stat[0, 1]

        loss = model.loss_fn(pred, target)

        eval_loss += loss.item()
    return eval_loss / len(eval_loader)

def eval(model, test_loader, cfg, output_dir, tb_logger=None, title=""):
    NLL_list = None
    RMSE_list = None

    gt_M_distance_list = []
    sample_M_distance_list = []


    with torch.no_grad():

        for cur_it, batch in enumerate(test_loader):
            input, target = batch
            # input, target, stat = batch # code for normalization new normalization
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization

            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    samples = model(input).tolist()
                    # out = stat[1, 1] * model(input) + stat[0, 1] # code for normalization new normalization
                    # samples = out.tolist()
                else:
                    model_output = model(input).tolist()
                    samples = np.append(samples, np.asarray(model_output), axis=1)
                    # out = stat[1, 1] * model(input) + stat[0, 1] # code for normalization new normalization
                    # out = out.tolist()
                    # samples = np.append(samples, np.asarray(out), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])

            NLL = evaluate_with_NLL(mean, var, target.tolist())  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=model(input).tolist(), mean=mean, var=var)
            sample_M_distance_list.extend(sample_M_distance)
            gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if NLL_list is None:
                NLL_list = NLL
                RMSE_list = RMSE
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))

            tb_logger.flush()

    print("NLL min/max: ", min(NLL_list), max(NLL_list))

    print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

    err_list = np.hstack([NLL_list.reshape(-1, 1), RMSE_list.reshape(-1, 1)])
    err_list = np.append(err_list,
                         np.asarray([[np.mean(NLL_list), np.mean(RMSE_list)],
                                     [np.std(NLL_list), np.std(RMSE_list)]]), axis=0)

    np.savetxt(os.path.join(output_dir, title + "_error_list.csv"), err_list, delimiter=",")

    # err_df = pd.DataFrame(err_list)
    # err_df.columns = ['NLL', "RMSE"]
    # err_df.append(pd.DataFrame(np.asarray([[np.mean(NLL_list), np.std(NLL_list)],
    #                                        [np.mean(RMSE_list), np.std(RMSE_list)]])))
    # err_df.to_csv(os.path.join(output_dir, title + "_error_list.csv"))

    plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter(NLL_list, RMSE_list, output_dir, title=title)
    plot_Mahalanobis_distance(sample_M_distance_list,gt_M_distance_list, output_dir=output_dir, title=title)
    return

def eval_batch(model, data):
    loss = model_fn(model, data)


# def compute_mean_and_variance(samples, num_networks):
#     """
#     Compute the approximation of mean and variance for given input by MC dropout
#
#     :param samples: for each input in batch, we draw (num_networks) samples from model [Batch_size, num_networks]
#     :param num_networks: Dropout approximate the ensemble. num_networks means the number of times we draw samples from our dropout model.
#     :return: approximated mean and variance
#     """
#
#     mean = np.mean(samples, axis=1) #shape(mean) = [batch_size, num_networks]
#
#     mean = np.reshape(mean, (-1,1))
#     #var =(np.sum(np.square(samples),axis=1))/num_networks - np.square(mean) # shape(var) = [Batch size, num_networks]
#     var = (np.sum(np.square(np.subtract(samples, mean)), axis=1))/num_networks
#
#     # check the values
#     #print('mean shape', np.shape(mean), mean)
#     #print('samples', samples[0:2])
#     #print('means', np.mean(var))
#     #print('std', np.std(var))
#
#     return np.reshape(mean,(-1, 1)), np.reshape(var,(-1,1))


def compute_test_loss(model: nn.Module, test_set_dir, batch_size, num_worker, device):
    """
    A function to compute the loss of model in training phase given the test dataset.

    :param model:
    :param test_set_dir:
    :param batch_size:
    :param dropout_prop:
    :param num_worker:
    :param input_dim:
    :param device: GPU or CPU
    :return:
    """
    print('Start Evaluating with MC Dropout')
    test_dataset = WineDataset(data_dir=test_set_dir,transform=None)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True, num_workers=num_worker)

    model = model.float()
    model = model.to(device)

    model.train()

    test_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input, label = batch

            # preprocessing inputs and labels in order to prevent some runtime error( i.e tensor dim or datatype not matching)
            input = input.to(device)
            input = input.float()
            label = label.to(device)
            label = label.float()
            label = label.view(-1,1) #Tensor reshape

            model_output = model(input)
            MSELoss = nn.MSELoss()
            loss = MSELoss(model_output, label)
            test_loss += loss

    test_loss = test_loss/len(test_dataloader)
    print("Test loss: ",test_loss)

    return test_loss


def evaluate_with_NLL(mean, var, label):
    epsilon = 1e-3
    var[var==0] =epsilon # replace where the value is zero to small number(epsilon) to prevent the operation being devided by zero
    var[var<epsilon] = epsilon
    NLL = np.log(var)*0.5 + np.divide(np.square(label-mean),(2*(var)))
    # NLL[NLL >= 50] = 50


    return NLL

def compute_mean_and_variance(samples, num_networks):
    """
    Compute the approximation of mean and variance for given input by MC dropout
    :param samples: for each input in batch, we draw (num_networks) samples from model [Batch_size, num_networks]
    :param num_networks: Dropout approximate the ensemble. num_networks means the number of times we draw samples from our dropout model.
    :return: approximated mean and variance
    """

    mean = np.mean(samples, axis=1) #shape(mean) = [batch_size, num_networks]
    # print('mean shape', np.shape(mean), mean)
    var =(np.sum(np.square(samples),axis=1))/num_networks - np.square(mean) # shape(var) = [Batch size, num_networks]

    return np.reshape(mean,(-1, 1)), np.reshape(var,(-1,1))

def evaluate_with_RMSE(mean, label):

    return np.sqrt(np.square(label - mean))

def assess_uncertainty_realism(gt_label, sample, mean, var):
    """
    compute the 1-dimension Mahalanobis distance given the ground truth label and sample drawn from the model

    :param gt_label: ground truth label
    :param sample: single sample drawn from MC dropout model
    :param mean: sample mean of 50 samples
    :param var: sample variance of 50 samples
    :return: list of computed sample Mahalanobis distance and gt Mahalanobis distance
    """

    sample_M_distance = np.divide(np.square(sample-mean),var) # computed Mahalanobis distance given sample, but since the model output is 1D, we compute 1D version of Mahalanobis distance
    gt_M_distance = np.divide(np.square(gt_label-mean),var) # computed Mahalanobis distance given ground truth label

    return (np.squeeze(sample_M_distance).tolist(), np.squeeze(gt_M_distance).tolist())