import argparse
import random
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

import np_transforms as NP_T
from datasets import WebcamTSeq
from model import *
from utils import show_images, sort_seqs_by_len
import plotter

def get_data_loaders(args_path, args_shape, train_transform, args_gamma, args_batch_size, file_name, args_max_len):
    train_data = WebcamTSeq(path=args_path, out_shape=args_shape, transform=train_transform, gamma=args_gamma, max_len=args_max_len, file_name=file_name)
    train_loader = DataLoader(train_data,
                            batch_size=args_batch_size,
                            shuffle=False)  # shuffle the data at the beginning of each epoch

    del train_data

    return train_loader, None

def main():
    parser = argparse.ArgumentParser(description='Train FCN-rLSTM in WebCamT dataset (sequential version).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', default='./model/fcn_bla_wct.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('-d', '--dataset', default='WebCamT', type=str, metavar='', help='dataset')
    parser.add_argument('-p', '--data_path', default='./data/WebCamT', type=str, metavar='', help='data directory path')
    parser.add_argument('--valid', default=0, type=float, metavar='', help='fraction of the training data for validation')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='', help='learning rate')
    parser.add_argument('--ct', default=False, type=bool, metavar='', help='continue training from a previous model')
    parser.add_argument('--epochs', default=200, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, metavar='', help='batch size')
    parser.add_argument('--img_shape', default=[120, 160], type=int, metavar='', help='shape of the input images')
    parser.add_argument('--lambda', default=1e-3, type=float, metavar='', help='trade-off between density estimation and vehicle count losses (see eq. 7 in the paper)')
    parser.add_argument('--gamma', default=1e3, type=float, metavar='', help='precision parameter of the Gaussian kernel (inverse of variance)')
    parser.add_argument('--max_len', default=5, type=int, metavar='', help='maximum sequence length')
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='weight decay regularization')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_tensorboard', default=True, type=int, metavar='', help='use TensorBoardX to visualize plots')
    parser.add_argument('--tb_img_shape', default=[120, 160], type=int, metavar='', help='shape of the images to be visualized in TensorBoardX')
    parser.add_argument('--log_dir', default='./log/fcn_rlstm_wct_train', help='tensorboard log directory')
    parser.add_argument('--n2show', default=2, type=int, metavar='', help='number of examples to show in Visdom in each epoch')
    args = vars(parser.parse_args())

    # dump args to a txt file for your records
    with open(args['model_path'] + '.txt', 'w') as f:
        f.write(str(args)+'\n')

    # if args['use_cuda'] == True and we have a GPU, use the GPU; otherwise, use the CPU
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    file_list = ['164', '166', '170_1', '170_2', '173_1', '173_2', '181', '253_1', '253_2', '398_1', '398_2',
                '403_1', '403_2', '410_1', '410_2', '495_1', '495_2', '511_1', '511_2', '551_1', '551_2',
                '572_1', '572_2', '691_1', '691_2', '846_1', 'bigbus', '846_2']

    # instantiate the model and define an optimizer
    if(args['ct']):
        model = FCN_BLA(FCN,Encoder,Decoder, image_dim=(torch.zeros(args['img_shape']))).to(device)
        model.load_state_dict(torch.load(args['model_path']))
        print("Existing model loaded")
    else:
        model = FCN_BLA(FCN,Encoder,Decoder, image_dim=(torch.zeros(args['img_shape'], dtype=torch.int32))).to(device)
        print("New model loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # Tensorboard is a tool to visualize plots during training
    if args['use_tensorboard']:
        tensorboard_plt = plotter.TensorboardPlotter(log_dir=args['log_dir'])
        args_str = '\n'.join(['{}={} | '.format(k, v) for k, v in args.items()])
        tensorboard_plt.text_plot("Train Args", args_str,0)
        tensorboard_plt.text_plot("Model Structure", str(model),0)
        tensorboard_plt.text_plot("Evaluation Method", "Global Loss = MSE, Density Loss = MSE, Count Loss = MSE, Count Error = MAE", 0)

    # for early stopping
    best_valid_loss = np.inf

    # training routine
    for epoch in range(args['epochs']):
        print('Epoch {}/{}'.format(epoch, args['epochs']-1))

        # training phase
        model.train()  # set model to training mode (affects batchnorm and dropout, if present)
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        X, mask, density, count = None, None, None, None
        t0 = time.time()

        for file_elem in file_list:
            train_loader, valid_loader = get_data_loaders(args_path=args['data_path'], args_shape=args['img_shape'], train_transform=NP_T.ToTensor(),
            args_gamma=args['gamma'], args_batch_size=args['batch_size'], file_name=file_elem, args_max_len=args['max_len'])

            print("WebCamT "+file_elem+" data loaded")

            for i, (X, mask, density, count, _, seq_len) in enumerate(train_loader):
                # copy the tensors to GPU (if applicable)
                X, mask, density, count= X.to(device), mask.to(device), density.to(device), count.to(device)

                # forward pass through the model
                density_pred, count_pred = model(X, mask=mask, lengths=seq_len)

                # compute the loss
                N = torch.sum(seq_len)
                density_loss = torch.sum((density_pred - density)**2)/(2*N)
                count_loss = torch.sum((count_pred - count[:,-1].unsqueeze(1))**2)/(2*args['batch_size'])
                loss = density_loss + args['lambda']*count_loss

                # backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('{}/{} mini-batch loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f}'
                    .format(i, len(train_loader)-1, loss.item(), density_loss.item(), count_loss.item()),
                    flush=True, end='\r')

                # save the loss values
                loss_hist.append(loss.item())
                density_loss_hist.append(density_loss.item())
                count_loss_hist.append(count_loss.item())
                with torch.no_grad():  # evaluation metric, so no need to compute gradients
                    count_err = torch.sum(torch.abs(count_pred - count[:,-1]))/args['batch_size']
                count_err_hist.append(count_err.item())
        t1 = time.time()
        print()

        # print the average training losses
        train_loss = sum(loss_hist)/len(loss_hist)
        train_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        train_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        train_count_err = sum(count_err_hist)/len(count_err_hist)

        print('Training statistics:')
        print('global loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
            .format(train_loss, train_density_loss, train_count_loss, train_count_err))
        print('time: {:.0f} seconds'.format(t1-t0))

        if args['use_tensorboard']:
            tensorboard_plt.loss_plot('Global Loss', 'train', train_loss, epoch)
            tensorboard_plt.loss_plot('Density Loss', 'train', train_density_loss, epoch)
            tensorboard_plt.loss_plot('Count Loss', 'train', train_count_loss, epoch)
            tensorboard_plt.loss_plot('Count Error', 'train', train_count_err, epoch)

            # show a few training examples (images + density maps)
            if epoch % 10 == 0:
                X *= mask  # show the active region only
                N, L, C, H, W = X.shape
                X, density, count = X.cpu().numpy(), density.cpu().numpy(), count.cpu().numpy()
                X = X[:,-1,:,:,:]
                density = density[:,-1,:,:]
                count = count[:,-1].reshape(N,1)
                density_pred, count_pred = density_pred.detach().cpu().numpy()[:,-1,:,:,:], count_pred.detach().cpu().numpy()
                n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
                show_images(tensorboard_plt, 'Ground Truth', 'train', X[0:n2show], density[0:n2show], count[0:n2show], shape=args['tb_img_shape'],global_step=epoch)
                show_images(tensorboard_plt, 'Prediction', 'train', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['tb_img_shape'],global_step=epoch)

        del train_loader, X, density, count
        
        if valid_loader is None:
            print()
            continue

        # validation phase
        model.eval()  # set model to evaluation mode (affects batchnorm and dropout, if present)
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        X, mask, density, count = None, None, None, None
        t0 = time.time()
        for i, (X, mask, density, count, _, seq_len) in enumerate(valid_loader):
            # copy the tensors to GPU (if applicable)
            X, mask, density, count= X.to(device), mask.to(device), density.to(device), count.to(device)
            # sort the sequences by descending order of the respective lengths (as expected by the model)
            seqs, seq_len = sort_seqs_by_len([X, mask, density, count], seq_len)
            X, mask, density, count = seqs

            # forward pass through the model
            with torch.no_grad():  # no need to compute gradients in validation (faster and uses less memory)
                density_pred, count_pred = model(X, mask=mask, lengths=seq_len)

            # compute the loss
            N = torch.sum(seq_len)
            density_loss = torch.sum((density_pred - density)**2)/(2*N)
            count_loss = torch.sum((count_pred - count)**2)/(2 * args['batch_size'])
            loss = density_loss + args['lambda']*count_loss

            # save the loss values
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            count_err = torch.sum(torch.abs(count_pred - count[:,-1].unsqueeze(1)))/args['batch_size']
            count_err_hist.append(count_err.item())
        t1 = time.time()
        # print the average validation losses
        valid_loss = sum(loss_hist)/len(loss_hist)
        valid_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        valid_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        valid_count_err = sum(count_err_hist)/len(count_err_hist)

        # scheduler step
        scheduler.step(valid_loss)

        print('Validation statistics:')
        print('global loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
            .format(valid_loss, valid_density_loss, valid_count_loss, valid_count_err))
        print('time: {:.0f} seconds'.format(t1-t0))
        print()

        if args['use_tensorboard']:
            # Single plot for all validation losses
            tensorboard_plt.loss_plot('Global Loss', 'valid', valid_loss, epoch)
            tensorboard_plt.loss_plot('Density Loss', 'valid', valid_density_loss, epoch)
            tensorboard_plt.loss_plot('Count Loss', 'valid', valid_count_loss, epoch)
            tensorboard_plt.loss_plot('Count Error', 'valid', valid_count_err, epoch)

            # Overlap plot for validation losses
            tensorboard_plt.overlap_plot('Global Loss',{'train':train_loss,'valid':valid_loss}, epoch)
            tensorboard_plt.overlap_plot('Density Loss',{'train':train_density_loss,'valid':valid_density_loss}, epoch)
            tensorboard_plt.overlap_plot('Count Loss',{'train':train_count_loss,'valid':valid_count_loss}, epoch)
            tensorboard_plt.overlap_plot('Count Error',{'train':train_count_err,'valid':valid_count_err}, epoch)

            # show a few training examples (images + density maps)
            if epoch % 10 == 0:
                X *= mask  # show the active region only
                N, L, C, H, W = X.shape
                X, density, count = X.cpu().numpy(), density.cpu().numpy(), count.cpu().numpy()
                X = X[:,-1,:,:,:]
                density = density[:,-1,:,:]
                count = count[:,-1].reshape(N,1)
                density_pred, count_pred = density_pred.detach().cpu().numpy()[:,-1,:,:,:], count_pred.detach().cpu().numpy()
                
                n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
                show_images(tensorboard_plt, 'Ground Truth', 'valid', X[0:n2show], density[0:n2show], count[0:n2show], shape=args['tb_img_shape'],global_step=epoch)
                show_images(tensorboard_plt, 'Prediction', 'valid', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['tb_img_shape'],global_step=epoch)
        
        # save the model if the validation loss is the best so far
        if valid_loss < best_valid_loss or epoch % 50 == 0:
            best_valid_loss = min(valid_loss, best_valid_loss)
            save_dir = args['model_path'].split('.pth')[0]
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), save_dir + '/model' + str(epoch) + '.pth')
            print('Model saved to {}'.format(save_dir + '/model' + str(epoch) + '.pth'))
        del valid_loader, X, density, count

    if args['use_tensorboard']:
        tensorboard_plt.close()

    torch.save(model.state_dict(), args['model_path'])


if __name__ == '__main__':
    main()
