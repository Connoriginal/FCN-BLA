from calendar import EPOCH
import numpy as np
import skimage.transform as SkT
import torch


def gauss2d(shape, center, gamma, out_shape=None):
    H, W = shape
    if out_shape is None:
        Ho = H
        Wo = W
    else:
        Ho, Wo = out_shape
    x, y = np.array(range(Wo)), np.array(range(Ho))
    x, y = np.meshgrid(x, y)
    x, y = x.astype(float)/Wo, y.astype(float)/Ho
    x0, y0 = float(center[0])/W, float(center[1])/H
    G = np.exp(-(1/2)*(((x - x0)*gamma[0])**2 + ((y - y0)*gamma[1])**2))  # Gaussian kernel centered in (x0, y0)
    return G/np.sum(G)  # normalized so it sums to 1

def density_map(shape, centers, gammas, out_shape=None):
    if out_shape is None:
        D = np.zeros(shape)
    else:
        D = np.zeros(out_shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), gammas[i], out_shape=out_shape)
    return D

def show_images(plt, tag, type, X, density, count, shape=None, global_step=0):
    labels = ['img {} count = {} | '.format(i, int(cnti)) for i, cnti in enumerate(count)]
    labels_str = '\n'.join(labels)
    if shape is not None:
        N = X.shape[0]  # N, C, H, W
        X, density = X.transpose(2, 3, 0, 1), density.transpose(2, 3, 0, 1)  # H, W, N, C (format expected by skimage)
        X, density = SkT.resize(X, (shape[0], shape[1], N, 3)), SkT.resize(density, (shape[0], shape[1], N, 1))
        X, density = X.transpose(2, 3, 0, 1), density.transpose(2, 3, 0, 1)  # N, C, H, W
    Xh = np.tile(np.mean(X, axis=1, keepdims=True), (1, 3, 1, 1))
    density = np.squeeze(density)
    density[density < 0] = 0.
    scale = np.max(density, axis=(1, 2))[:, np.newaxis, np.newaxis] + 1e-9
    density /= scale
    Xh[:, 1, :, :] *= 1 - density
    Xh[:, 2, :, :] *= 1 - density
    density = np.tile(density[:, np.newaxis, :, :], (1, 3, 1, 1))

    ## Tensorboard Version  ## => tensorboard add_image는 4차원이면 Tensor로 변환 필수? tensorboard는 이미지 한개만 허용 뭔가 매번 싯팔 학습때마다 돌려야할것 같은 느낌이 드는데 기분 탓이길 바람ㅎㅎ
    plt.img_plot(tag + ' Highlighted' + '/' + type, torch.Tensor(Xh), global_step)
    plt.text_plot(tag + ' Highlighted' + '/' + type, labels_str, global_step)
    plt.img_plot(tag + ' Density Maps' + '/' + type, torch.Tensor(density), global_step)
    plt.text_plot(tag + ' Density Maps' + '/' + type, labels_str, global_step)

def sort_seqs_by_len(seqs, seq_len):
    sort_idx = torch.argsort(seq_len, dim=0, descending=True)
    seq_len_sort = seq_len[sort_idx]
    seqs_sort = []
    for seq in seqs:
        seqs_sort.append(seq[sort_idx])

    return seqs_sort, seq_len_sort
