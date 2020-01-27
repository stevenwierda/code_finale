from __future__ import print_function, division
import os

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tr
from utils.npy_crop import crop, get_params
import torch
from utils import errorclasses as err
import cv2
from utils import functions as func
""""
Lists of paths
Load the data with numpy and cv2
Images are npy files 
Masks/Labels are png files, added together
"""

class MaterialDataset(Dataset):
    """
    path_to_data: path to the images and masks folders. It is initialized in train.py
    mode: train or val or test
    """

    def __init__(self, args, mode, transforms=None, module = None):
        """Initialization"""
        self.args = args
        self.crop_size = args.crop_size
        self.subsamples = args.subsamples if mode == 'train' else 1
        self.img_path = os.path.join(self.args.dataset, args.imgs)
        self.mask_path = os.path.join(self.args.dataset, args.labels)

        self.selec_img = []
        self.selec_mask = []

        if transforms is None:
            self.transforms = tr.Compose([tr.ToTensor()]) #default transform
        else:
            self.transforms = transforms

        if isinstance(mode, str):
            self.mode = [mode]
        else:
            raise Exception('mode argument must be a string (train, val or pred.txt)')

        self.module = module

        with open(os.path.join(args.dataset_txt, self.mode[0] + '.txt'),'r') as file:
            lines = file.read().splitlines() #list of the txt file's lines
        for img_name in lines:
            impath = os.path.join(self.img_path, img_name + ".npy")
            mpath = os.path.join(self.mask_path, img_name + ".png")

            self.selec_img.append(impath)
            self.selec_mask.append(mpath)

        if len(self.selec_img) == 0:
            raise err.EmptyFolderError("No data npy selected.")
        if len(self.selec_mask) == 0:
            raise err.EmptyFolderError("No mask png selected.")

        assert (len(self.selec_img) == len(self.selec_mask))
        print("\nNumber of images to be processed in {} mode is: {:d}".format(self.mode[0], len(self.selec_img)))

    def load_mask(self, index):
        #HxWxC
        print(self.selec_mask[index])
        mask = cv2.cvtColor(cv2.imread(self.selec_mask[index]), cv2.COLOR_BGR2RGB)
        print("this is the mask shape",mask.shape)
        print(self.selec_mask[index])
        mask = func.encode_segmap(mask)
        #mask = np.load(self.selec_mask[index])
        #mask = np.transpose(mask,[2,0,1])
        return mask

    def load_img(self, index):
        # In Plastic dataset: HxWxC ---> transpose to CxHxW
        #img = np.transpose(np.load(self.selec_img[index]),[2,0,1])
        img = np.load(self.selec_img[index])
        print("this is the shape of the image", img.shape)
        return img

    def __getitem__(self, index):
        #img = self.transforms(self.load_img(index)).float()
        #target = self.transforms(self.load_mask(index)).float()
        #img transpose CxHxW <-- HxWxC
        img = torch.as_tensor(np.transpose(self.load_img(index), [2,0,1]), dtype = torch.float)
        if self.args.sampler == "classic" or self.mode[0] == "val":
            target = torch.as_tensor(self.load_mask(index), dtype=torch.long)
            if self.args.crop_size is not None and self.subsamples > 1:
                batch = list()
                for sub in range(self.subsamples):
                    batch.append(crop(img, target, get_params(img.shape[1], img.shape[2], self.crop_size)))
                return batch
            else:
                if self.module is not None:
                    height_offset, width_offset = img.shape[1] % self.module, img.shape[2] % self.module
                    img = img[:, height_offset:,width_offset:]
                    target = target[height_offset:, width_offset:]
                    return img, target
        elif self.args.sampler == "tile":
            target = self.load_mask(index)
            img, target, check = func._tile_segmap_v2(img, target, self.args.nrtiles, self.crop_size)
            target = torch.as_tensor(target,dtype=torch.long)
            if np.max(check) == 0:
                raise("something is wrong with the loading of the tiles")
            batch = list()
            for i in range(img.shape[0]):

                batch.append((img[i], target[i]))
            return batch

        else:
            raise err.WrongArgumentError("No valid mode selected")
        return img, target

    def __len__(self):
        return len(self.selec_img) #*nb of patches