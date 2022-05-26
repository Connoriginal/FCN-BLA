import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import np_transforms as NP_T
from datasets import TrancosSeq
from model import *
from utils import show_images, sort_seqs_by_len
import plotter

def main():
    parser = argparse.ArgumentParser(description='Test FCN-rLSTM in Trancos dataset (sequential version).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', default='./model/fcn_bla_tcs.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('-d', '--data_path', default='./data/TRANCOS_v3', type=str, metavar='', help='data directory path')
    parser.add_argument('--batch_size', default=32, type=int, metavar='', help='batch size')
    parser.add_argument('--img_shape', default=[120, 160], type=int, metavar='', help='shape of the input images')
    parser.add_argument('--gamma', default=1e3, type=float, metavar='', help='precision parameter of the Gaussian kernel (inverse of variance)')
    parser.add_argument('--max_len', default=5, type=int, metavar='', help='maximum sequence length')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_tensorboard', default=True, type=int, metavar='', help='use TensorBoardX to visualize plots')
    parser.add_argument('--log_dir', default='./log/fcn_bla_test', help='tensorboard log directory')
    parser.add_argument('--tb_img_shape', default=[120, 160], type=int, metavar='', help='shape of the images to be visualized in TensorBoardX')
    parser.add_argument('--n2show', default=8, type=int, metavar='', help='number of examples to show in Tensorboard in each epoch')
    parser.add_argument('--seed', default=-1, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # use a fixed random seed for reproducibility purposes
    if args['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])

    # if args['use_cuda'] == True and we have a GPU, use the GPU; otherwise, use the CPU
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    print('device:', device)

    # instantiate the dataset
    test_data = TrancosSeq(
        train=False,
        path=args['data_path'],
        out_shape=args['img_shape'],
        transform=NP_T.ToTensor(),
        gamma=args['gamma'],
        max_len=args['max_len'])

    # create a data loader
    test_loader = DataLoader(
        test_data,
        batch_size=args['batch_size'],
        shuffle=False)

    # instantiate the model
    model = FCN_BLA(FCN,Encoder,Decoder, image_dim=(test_data[0][0].shape[2:])).to(device)
    model.load_state_dict(torch.load(args['model_path'], map_location=device))
    print("model loaded")

    # Tensorboard is a tool to visualize plots during training
    if args['use_tensorboard']:
        tensorboard_plt = plotter.TensorboardPlotter(log_dir=args['log_dir'])
        samples = {'X': [], 'density': [], 'count': [], 'density_pred': [], 'count_pred': []}
        nsaved = 0

    # do inference and print statistics
    model.eval()  # set model to evaluation mode
    density_loss_hist = []
    count_loss_hist = []
    count_err_hist = []
    t0 = time.time()
    for i, (X, mask, density, count, _, seq_len) in enumerate(test_loader):
        # copy the tensors to GPU (if applicable)
        X, mask, density, count = X.to(device), mask.to(device), density.to(device), count.to(device)
        # sort the sequences by descending order of the respective lengths (as expected by the model)
        seqs, seq_len = sort_seqs_by_len([X, mask, density, count], seq_len)
        X, mask, density, count = seqs

        # forward pass through the model
        with torch.no_grad():  # no need to compute gradients in test (faster and uses less memory)
            density_pred, count_pred = model(X, mask=mask)

        # compute the performance metrics
        N = torch.sum(seq_len)
        density_loss = torch.sum((density_pred - density)**2)/(2*N)
        count_loss = torch.sum((count_pred - count[:,-1].unsqueeze(1))**2)/(2 * args['batch_size'])
        count_err = torch.sum(torch.abs(count_pred - count[:,-1].unsqueeze(1)))/(args['batch_size'])

        # save the results
        density_loss_hist.append(density_loss.item())
        count_loss_hist.append(count_loss.item())
        count_err_hist.append(count_err.item())
        
        # save a few examples to show in Tensorboard
        if args['use_tensorboard'] and (nsaved < args['n2show']):
            X *= mask
            N, L, C, H, W = X.shape
            X, density, count = X.cpu().numpy(), density.cpu().numpy(), count.cpu().numpy()
            X = X[:,-1,:,:,:]  # show the last frame only
            density = density[:,-1,:,:,:]
            count = count[:,-1].reshape(N,1)
            density_pred = density_pred.detach().cpu().numpy()[:,-1,:,:,:]
            count_pred = count_pred.detach().cpu().numpy()

            n2save = min(X.shape[0], args['n2show'] - nsaved)
            samples['X'].append(X[0:n2save])
            samples['density'].append(density[0:n2save])
            samples['count'].append(count[0:n2save])
            samples['density_pred'].append(density_pred[0:n2save])
            samples['count_pred'].append(count_pred[0:n2save])
            nsaved += n2save

        print('Testing... ({:.0f}% done)'.format(100.*(i+1)/len(test_loader)),
              flush=True, end='\r')
    print()
    density_loss = sum(density_loss_hist)/len(density_loss_hist)
    count_loss = sum(count_loss_hist)/len(count_loss_hist)
    count_err = sum(count_err_hist)/len(count_err_hist)
    t1 = time.time()

    print('Test statistics:')
    print('density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
          .format(density_loss, count_loss, count_err))
    print('time: {:.0f} seconds'.format(t1-t0))

    # show a few examples
    if args['use_tensorboard']:
        for key in samples:
            samples[key] = np.concatenate(samples[key], axis=0)
        
        show_images(tensorboard_plt, 'Ground Truth', 'test', samples['X'], samples['density'], samples['count'], shape=args['tb_img_shape'])
        show_images(tensorboard_plt, 'Prediction', 'test', samples['X'], samples['density_pred'], samples['count_pred'], shape=args['tb_img_shape'])
        tensorboard_plt.close()
        
if __name__ == '__main__':
    main()
