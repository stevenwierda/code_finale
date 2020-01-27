import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tr
from utils.functions import get_labels, decode_seg_map_sequence
from Aeroscan.unet import UNet
from Aeroscan.deepresunet import DeepResUNet
from Aeroscan.deeplab_components.deeplab import DeepLab
from Aeroscan.unetpp import UNetPP
from Aeroscan.dataloaders.dataloader import MaterialDataset
from train import parse_args

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)

args = parse_args()

class BuildingDataset(Dataset):
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

        self.selec_img = []

        if transforms is None:
            self.transforms = tr.Compose([tr.ToTensor()]) #default transform
        else:
            self.transforms = transforms

        if isinstance(mode, str):
            self.mode = [mode]
        else:
            raise Exception('mode argument must be a string (train, val or pred)')

        self.module = module

        with open(os.path.join(args.dataset_txt, self.mode[0] + '.txt'),'r') as file:
            lines = file.read().splitlines() #list of the txt file's lines
        for img_name in lines:
            impath = os.path.join(self.img_path, img_name + ".npy")

            self.selec_img.append(impath)

        print("\nNumber of images to be processed in {} mode is: {:d}".format(self.mode[0], len(self.selec_img)))

    def load_img(self, index):
        img = np.load(self.selec_img[index])
        return img

    def __getitem__(self, index):
        img = torch.as_tensor(np.transpose(self.load_img(index), [2,0,1]), dtype = torch.float)

        # clip the pixel values between mean-4*sd, mean+4*sd
        mean_channel = torch.mean(img, dim=(1,2))
        std_channel = torch.std(img,dim=(1,2))

        for c in range(img.shape[0]):
            min = mean_channel[c].item()-4*std_channel[c].item()
            max = mean_channel[c].item()+4*std_channel[c].item()
            img[c, :,:] = torch.clamp(img[c, :, :], min, max)

            if self.module is not None:
                height_offset, width_offset = img.shape[1] % self.module, img.shape[2] % self.module
                img = img[:, height_offset:,width_offset:]
            return img

    def __len__(self):
        return len(self.selec_img) #*nb of patches

def create_dataloaders(args, module):

    test_set = BuildingDataset(args, 'pred', module = module)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


    return test_loader

def create_network(args):

    if args.network == "UNet":
        network = UNet(in_channels=222, num_classes=args.num_classes)
        module = 2**2  # 2^n where n nb of convolutions
    elif args.network == "DeepResUNet":
        network = DeepResUNet(in_channels=222, out_classes=args.num_classes)
        module = 2**5
    elif args.network == "UNetPP":
        network = UNetPP(n_channels=222, n_classes=args.num_classes)
        module = 2**5
    elif args.network == "DeepLab":
        network = DeepLab(num_classes=args.num_classes)
        module = 2**5  # xception
    else:
        raise Exception('Unknown network')
    return network, module

def decode_label(args, input):
    n_classes = args.num_classes
    label_colours = get_labels()
    label = np.zeros((input.shape[1], input.shape[2],3))
    for i in range(n_classes):
        label[input[0]==i] = label_colours[i]
    return label

def prediction(args, action):

    # Creating network
    print("Creating network...")
    network, module = create_network(args)
    network.eval()
    # Loading the test set
    print("Creating dataloader...")
    test_loader = create_dataloaders(args, module)

    # Loading the model
    print("Loading the model...")
    dir = "/home/student/aeroscan/tileimp/run"

    if action == 'predict':
        #listnet=['UNetPP-tilev2']
        #for net in listnet:
            net=UNet
            print("Predicting with network", net)
            filename = os.path.join(dir, 'experiment_UNet_tile_hues2', 'checkpoint.pth.tar')
            model = torch.load(filename)
            print(model['epoch'], model['best_pred'])
            network.load_state_dict(model['state_dict'])
            network.cuda()

            print("Fetching data...")
            for i, img in enumerate(test_loader):
                # Get an output from the network
                print("Predicting...")
                with torch.no_grad():
                    img = img.to(device)
                    output = network(img)
                pred = output.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                decode_pred = decode_label(args, pred)
                print("Saving prediction no.", i + 1)
                np.save('/home/student/aeroscan/tileimp/predictions/pred_UNet_Tile_hue', decode_pred)

    elif action == 'detect_damage':
        startdirectory= "/home/student/aeroscan/tileimp/outputs/"
        Networkname = "deepresunet_Wood_trained_07_15-13-26"
        directory = os.path.join(startdirectory, Networkname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        net = UNetPP
        print("damage detection with network", net)
        filename = os.path.join(dir, 'experiment_DeepResUNet_identify_paint_fx10', 'checkpoint.pth.tar')
        model = torch.load(filename)
        print(model['epoch'], model['best_pred'])
        network.load_state_dict(model['state_dict'])
        network.cuda()

        print("Fetching data...")
        for i, img in enumerate(test_loader):
            # Get an output from the network
            print("Predicting...")
            with torch.no_grad():
                img = img.to(device)
                output = network(img)
            pred = output.data.cpu().numpy()
            prob = torch.as_tensor(pred, dtype = torch.float32)
            softmax = torch.softmax(prob, dim=1)
            softmax = softmax*100
            np.save(os.path.join(directory,"classchannel"), softmax)
            softmax = np.nanmax(softmax, axis=1)
            softmax = np.transpose(softmax, (1, 2, 0))
            np.save(os.path.join(directory,"probability1channel"),softmax)
            pred=np.argmax(pred, axis=1)
            decode_pred = decode_label(args, pred)
            np.save(os.path.join(directory,"decode"), decode_pred)
def main():
    action = 'predict'
    prediction(args, action)



if __name__ == "__main__":
    main()