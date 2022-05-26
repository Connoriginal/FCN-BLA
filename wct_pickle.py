import os
import argparse

import pickle
import gzip

import xml.etree.ElementTree as ET

from PIL import Image
from skimage import io
import numpy as np
import skimage.transform as SkT
import torch

from utils import density_map


## mk_bndboxes 함수 안에 들어가는 path는 './data/WebCamT'로 설정되어야지 './data/WebCamT' 폴더 안에 pickle 파일이 생성됨
def mk_bndboxes(path, image_files):
    bndboxes = {img_f: [] for img_f in image_files}
    for img_f in image_files:
        # open an xml file and find '&' and remove it (it is not a valid XML character)
        xml_file = open(os.path.join(path, img_f.replace('.jpg', '.xml')), 'r')
        xml_str = xml_file.read()
        xml_str = xml_str.replace('&', '')
        root = ET.fromstring(xml_str)
        for vehicle in root.iter('vehicle'):
            xmin = int(vehicle.find('bndbox').find('xmin').text)
            ymin = int(vehicle.find('bndbox').find('ymin').text)
            xmax = int(vehicle.find('bndbox').find('xmax').text)
            ymax = int(vehicle.find('bndbox').find('ymax').text)
            bndboxes[img_f].append((xmin, ymin, xmax, ymax))
            
    # pickle로 data 저장
    with gzip.open(path+'/vehicle_pixel_info.pickle', 'wb') as f:
        pickle.dump(bndboxes, f)

def load_example(img_f, bndboxes, out_shape, gamma, path):
        X = io.imread(os.path.join(path, img_f))
        mask_f = os.path.join(img_f.split(os.sep)[0], img_f.split(os.sep)[1])+'_msk.png'
        mask = Image.open(os.path.join(path, mask_f)).convert('L')
        mask = np.array(mask)
        mask = mask[:, :, np.newaxis].astype('float32')

        # device = 'cuda:0'
        # X, mask = torch.from_numpy(X).to(device), torch.from_numpy(mask).to(device)

        H_orig, W_orig = X.shape[0], X.shape[1]
        # reduce the size of image and mask by the given amount
        H_orig, W_orig = X.shape[0], X.shape[1]
        if H_orig != out_shape[0] or W_orig != out_shape[1]:
            X = SkT.resize(X, out_shape, preserve_range=True).astype('uint8')
            mask = SkT.resize(mask, out_shape, preserve_range=True).astype('float32')

        # compute the density map
        img_centers = [(int((xmin + xmax)/2.), int((ymin + ymax)/2.)) for xmin, ymin, xmax, ymax in bndboxes]
        gammas = gamma*np.array([[1./np.absolute(xmax - xmin+0.001), 1./np.absolute(ymax - ymin+0.001)] for xmin, ymin, xmax, ymax in bndboxes])
        # gammas = self.gamma*np.ones((len(bndboxes), 2))
        density = density_map(
            (H_orig, W_orig),
            img_centers,
            gammas,
            out_shape=out_shape)
        density = density[:, :, np.newaxis].astype('float32')

        # return X.detach().cpu().numpy(), mask.detach().cpu().numpy(), density
        return X, mask, density

def main():
    parser = argparse.ArgumentParser(description='Save WebCamT Dataset to .pickle format', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', default='./data/WebCamT', type=str, metavar='', help='WebCamT Dataset path')
    parser.add_argument('--img_shape', default=[120,160], type=int, metavar='', help='shape of the input images')
    parser.add_argument('--gamma', default=1e3, type=float, metavar='', help='precision parameter of the Gaussian kernel (inverse of variance)')
    args = vars(parser.parse_args())
    
    image_files = []
    path=args['path']

    if not os.path.isfile(path+'/vehicle_pixel_info.pickle'):
        for cam in os.listdir(path):
            if not os.path.isdir(os.path.join(path, cam)):
                continue

            for seq in os.listdir(os.path.join(path, cam)):
                if not os.path.isdir(os.path.join(path, cam, seq)):
                    continue

                if 'big_bus' in seq:
                    continue

                image_files.extend([os.path.join(cam, seq, f) for f in os.listdir(os.path.join(path, cam, seq)) if f[-4:] == '.jpg'])
    else:
        with gzip.open(path+'/vehicle_pixel_info.pickle', 'rb') as f:
            data = pickle.load(f)

        image_files = list(data.keys())

    image_files.sort()

    # vehicle_pixel_info.pickle이 없을 경우 생성
    if not os.path.isfile(path+'/vehicle_pixel_info.pickle'):
        mk_bndboxes(path, image_files)
    with gzip.open(path+'/vehicle_pixel_info.pickle','rb') as f:
        bndboxes = pickle.load(f)

    # 폴더 별로 pickle 파일 생성
    max_flag = ['164/164-20160223-14/000398.jpg', '166/166-20160223-15/000494.jpg',
                '170/170-20160704-18/000292.jpg', '173/173-20160704-18/000300.jpg',
                '181/181-20151224-15/000500.jpg', '253/253-20160704-15/000292.jpg',
                '398/398-20160704-18/000299.jpg', '403/403-20160508-18/000298.jpg',
                '410/410-20160704-18/000292.jpg', '495/495-20160704-18/000299.jpg',
                '511/511-20160704-18/000293.jpg', '551/551-20160504-18/000500.jpg',
                '572/572-20160223-13/000499.jpg', '691/691-20160704-18/000294.jpg',
                '846/846-20160704-18/000300.jpg', '928/928-20160704-18/000293.jpg',
                'bigbus/bigbus-511/000115.jpg']
    subset_flag = ['170', '173', '253', '398', '403', '410', '495', '511', '551', '572', '691', '846']

    images, masks, densities = [], [], []
    for img_f in image_files:
        file_name = img_f.split(os.sep)[0]
        if os.path.isfile(path+'/'+file_name+'.pickle'):
            continue
        print(img_f)
        X, mask, density = load_example(img_f=img_f, bndboxes=bndboxes[img_f], out_shape=args['img_shape'], gamma=args['gamma'], path=path)
        images.append(X)
        masks.append(mask)
        densities.append(density)

        if img_f in max_flag:
            if img_f.split(os.sep)[0] in subset_flag:
                half = int(len(images)/2)
                with gzip.open(path+'/'+file_name+'_1.pickle', 'wb') as f:
                    pickle.dump({'images':images[:half], 'masks':masks[:half], 'densities':densities[:half]}, f)
                with gzip.open(path+'/'+file_name+'_2.pickle', 'wb') as f:
                    pickle.dump({'images':images[half:], 'masks':masks[half:], 'densities':densities[half:]}, f)
            else:
                with gzip.open(path+'/'+file_name+'.pickle', 'wb') as f:
                    pickle.dump({'images':images, 'masks':masks, 'densities':densities}, f)

            del images, masks, densities
            images, masks, densities = [], [], []

if __name__ == '__main__':
    main()