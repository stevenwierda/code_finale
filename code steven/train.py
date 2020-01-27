from __future__ import print_function, division
import os
import random

import argparse
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.saver import Saver
from utils.summaries import TensorboardSummary

import utils.errorclasses as err

from Aeroscan.unet import UNet
from Aeroscan.deepresunet import DeepResUNet
from Aeroscan.unetpp import UNetPP
from Aeroscan.deeplab_components.deeplab import DeepLab
from Aeroscan.dataloaders.dataloader import MaterialDataset
from utils.metrics import Evaluator
from utils.radam import RAdam

#Paths to data
#path_to_data = "/home/student/aeroscan/FromScratch1/Aeroscan/dataloaders/datasets"
#path_to_img = "/media/public_data/Projects/intern/Plastics/DataSet-PET-PEF/Dataset"


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Set the seed value, value to change
# seed_value = random.seed(20)

def parse_args():
    """
    This function adds arguments for the command-line.
    This allows to change the parameters
    """
    parser = argparse.ArgumentParser(description = "Training Script Pytorch Project Aeroscan")
    # Seed value
    parser.add_argument("--seed_value", type=int, help="Initialisation of the seed value", default=random.seed())
    # Network parameters
    parser.add_argument("-n","--network", type = str, help = "Network",
                        choices = ["UNet", "DeepResUNet", "UNetPP", "DeepLab", "MRDnet"],default = "UNet")
    parser.add_argument("-bb","--backbone", type = str, help = "Backbone for DeepLab",
                        choices = ["drn","mobilenet","resnet","xception"], default = 'xception')
    parser.add_argument("--sampler", help = "selects the classic or tiles mode", choices = ["tile", "classic"], default = "classic")

    # Training parameters
    parser.add_argument("-e", "--epochs", type = int, help = "The number of epochs", default = 100)
    parser.add_argument("--start_epoch", type = int, help = "Starting epoch", default = 9)
    parser.add_argument("-bs", "--batch_size", type = int, help = "Number of samples processed for each split", default = 2)
    parser.add_argument("-lr", "--learning_rate", type = float, help = "Learning rate", default = 0.0001)
    parser.add_argument("-m", "--momentum", type = float, help = "Momentum", default = 1e-3)
    parser.add_argument("-numw", "--num_workers", type = int, help = "Number of workers", default = 0)
    parser.add_argument("--optimizer", help = "Optimizer", choices = ("adam", "radam", "sgd", "nesterov"), default="adam")
    parser.add_argument("--weightdecay", help = "Weight decay of the optimizer", default = 0, type = float)

    # Resume
    #parser.add_argument("--resume", help = "Resume from a checkpoint", default= "run/experiment_35/checkpoint.pth.tar", type=str)
    parser.add_argument("--resume", help = "Resume from a checkpoint", default= None, type=str)

    # Scheduler parameters
    parser.add_argument("--step", type = int, help = "Interval between each lr reduction", default=10)
    parser.add_argument("--gamma", type=float, help = "Multiplying coef", default=0.1)

    # Sampling parameters
    parser.add_argument("--crop_size",
                        help = "Crop size for subsampling. Resulting img will be a square.",
                        default = 128, type = int)
    parser.add_argument("--subsamples", help = "Number of subsamples", type = int, default = 50)

    # Validation parameters
    parser.add_argument("--inter_eval", help="Interval between each evaluation", default=1,type=int)
    parser.add_argument("--num_classes", help="number of classes", default=10, type=int)
    #parser.add_argument("--snapshot_prefix", help="Snapshot prefix", default="snap/snapshot-")

    # Data
    parser.add_argument("--dataset_txt", help="Path to your txt files where the sets are defined", type=str,
                        default="/home/student/aeroscan/tileimp/Aeroscan/dataloaders/datasets/hues")
    parser.add_argument("--dataset", help="Dataset path",type=str,
                        default="/media/public_data/aeroscan/dataset/buildings_dataset/")
    parser.add_argument("--imgs", help=" prefix data folder", default="alldata/hues32")
    parser.add_argument("--labels", help="prefix labels folder", default="groundtruth")
    parser.add_argument("--nr_inchannels", help="gives the amount of in channels", type=int, default=222)

    #tiles
    parser.add_argument("--nrtiles", help="number of tiles created", type=int, default=50)


    args = parser.parse_args()
    return args

def create_dataloaders(args, module):
    train_set = MaterialDataset(args, 'train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
    val_set = MaterialDataset(args, 'val', module = module)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    return train_loader, val_loader, train_set, val_set,  # test_loader



def create_network(args):
    "selects the network and gives the amount of inchannels and classes"
    if args.network == "UNet":
        network = UNet(in_channels=args.nr_inchannels, num_classes=args.num_classes)
        module = 2**2  # 2^n where n nb of convolutions
    elif args.network == "DeepResUNet":
        network = DeepResUNet(in_channels=args.nr_inchannels, out_classes=args.num_classes)
        module = 2 ** 5
    elif args.network == "UNetPP":
        network = UNetPP(n_channels=args.nr_inchannels, n_classes=args.num_classes)
        module = 2**5
    elif args.network == "DeepLab":
        network = DeepLab(num_classes=args.num_classes, n_channels= args.nr_inchannels)
        module = 2**5  # xception
    else:
        raise Exception('Unknown network')
    return network, module


def optimizer_choice(args, network):
    "selects the optimizer"
    if args.optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weightdecay)
    elif args.optimizer == "radam":
        optimizer = RAdam(network.parameters(), lr=args.learning_rate)
    elif args.optimizer == "nesterov":
        optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weightdecay, nesterov=True)
    else:
        raise Exception("Unknown optimizer: {}".format(args.optimizer))
    return optimizer


def training(args, train_loader, train_set, network, loss, optimizer, writer, epoch):
    network.train()
    network.cuda()
    mean_loss = 0
    for i, batch in enumerate(train_loader):   #load the batches from the trainloader
        print("""\n>>>>>>> Batch number {}/{}""".format(i+1, len(train_loader)))
        sub_loss = 0.0
        for sub, sub_sample in enumerate(batch): #load the subsamples form the batches
            sub_img, sub_target = sub_sample
            sub_img = sub_img.to(device)
            sub_target = sub_target.to(device)

            # Get an output from the network
            output = network(sub_img)

            # Sets a full gradient of zeros RESEARCH
            network.zero_grad()
            # Get the loss between from the output and the target
            _loss = loss(output, sub_target)
            sub_loss += _loss.item()
            # Backward propagation through parameters
            _loss.backward()

            # Adjust weights through optimizer
            optimizer.step()

            print(">>>>>>>>>> Sub sample {}/{}".format(sub+1,len(batch)))
        sample_loss = sub_loss/args.nrtiles
        mean_loss += sample_loss
        writer.add_scalar('train/mean_loss', mean_loss, epoch)
        #batch = None

    #train_loader = None
    epoch_loss = mean_loss / train_set.__len__()
    print("Epoch mean loss: {} {}".format(epoch_loss, mean_loss))

def validation(args, val_loader, network, epoch, loss, optimizer, writer, saver, summary, best_pred, classnames):

    # Define Evaluator
    evaluator = Evaluator(args.num_classes)

    test_loss = 0.0
    evaluator.reset()

    #Put the model in eval() mode
    network.eval()
    network.cuda()
    num_img_tr = len(val_loader)
    for i, sample in enumerate(val_loader):
        # Get an output from the network
        img, target = sample
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            output = network(img)
        test_loss += loss(output, target)
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)
        if i % (num_img_tr // 1) == 0:
            global_step = i + num_img_tr * epoch
            summary.visualize_image(writer, img, target, pred, global_step)
        # sample, img, target, output, prediction = None, None, None, None, None
    val_loader = None
        #val_loader = None

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    F1score = evaluator.F1_Score()
    Recall = evaluator.Recall()
    Precision = evaluator.Precision()
    IoU = evaluator.IoU()
    evaluator.plot_confusion_matrix(evaluator.confusion_matrix,
                                    classes=classnames,
                                    normalize=True)

    # Send the confusion matrix to TensorBoard
    confusion_matrix = plt.imread('/home/student/aeroscan/tileimp/test.png')[:,:,:3]
    confusion_matrix = np.transpose(confusion_matrix, (2,0,1))
    confusion_matrix = torch.from_numpy(confusion_matrix)
    writer.add_image('confusion matrix', confusion_matrix, global_step)

    writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
    writer.add_scalar('val/mIoU', mIoU, epoch)
    writer.add_scalar('val/Acc', Acc, epoch)
    writer.add_scalar('val/Acc_class', Acc_class, epoch)
    writer.add_scalar('val/fwIoU', FWIoU, epoch)
    writer.add_scalar("val/F1score", F1score, epoch)
    writer.add_scalar("val/Recall", Recall, epoch)
    writer.add_scalar("val/Precision", Precision, epoch)
    writer.add_scalar("val/IoU", Precision, epoch)
    print('Validation:')
    print('[Epoch:', epoch)
    print("""
            Acc:{}          Acc_class:{}
            mIoU:{}         fwIoU:{}
            Precision:{}    F1score:{}
            Recall:{}       IoU:{}"""
            .format(Acc,Acc_class,mIoU,FWIoU,Precision,F1score,Recall,IoU))
    print('Loss: %.3f' % test_loss)

    new_pred = Precision
    if new_pred > best_pred:
        print("New best pred.txt.")
        is_best = True
        best_pred = new_pred
        saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best)
    return best_pred, saver



def main(args):

    deb = time()

    # Create network
    print("Creating network...")
    network, module = create_network(args)
    #classlist = ['BG', 'door', 'blind', 'windowframe', 'walls', 'drainpipe', 'windowborder', 'chimney', 'unknown', 'rooftop', 'metalfend']
    classlist = ['BG', 'wijzonol', 'sigma', 'sikkens', 'trimetal']
    #classlist = ['BG', 'sample1', 'sample2', 'sample3']
    # Create dataloaders
    print("Creating data loaders...")

    train_loader, val_loader, train_set, _, = create_dataloaders(args, module)

    # Loss creation, initialization
    print("Initializing loss function...")
    loss = CrossEntropyLoss()

    # Optimizer definition
    print("Defining optimizer and scheduler...")
    optimizer = optimizer_choice(args, network)
    # Scheduler definition
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step, gamma=args.gamma)

    # Define Saver
    saver = Saver(args)
    saver.save_experiment_config()

    # Define Tensorboard Summary
    summary = TensorboardSummary(saver.experiment_dir, args)
    writer = summary.create_summary()

    # Resume
    best_pred = 0.0
    if args.resume is not None:
        print("Loading snapshot of network")
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint['state_dict'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    # Train and evaluate network
    print("""
                    ********* Training starts *********
        Network:            {}           Loss Function:  CrossEntropy
        Optimizer:          {}
        Number of epochs:   {}
        Batch Size:         {}                     
        Learning rate:      {}              Momentum:   {}"""
          .format(args.network, args.optimizer, args.epochs,
                  args.batch_size, args.learning_rate, args.momentum))
    progress = tqdm(range(1, args.epochs +1))
    for epoch in progress:
        start = time()
        training(args, train_loader, train_set, network, loss, optimizer, writer, epoch)
        time_train = time()
        print("Training phase for epoch {} lasts {} sec.".format(epoch, time_train-start))
        if (epoch+1) % args.inter_eval == 0:
            print('-'*50, 'Evaluation', '-'*50)
            best_pred, _ = validation(args, val_loader, network, epoch, loss, optimizer, writer, saver, summary, best_pred, classlist)
            time_val = time()
            print("Validation phase for epoch {} lasts {} sec.".format(epoch, time_val - time_train))

        # Scheduler
        scheduler.step()

        progress.set_description("Epoch {}/{}".format(epoch, args.epochs,loss))
        progress.refresh()

    fin = time()
    min = (fin-deb)/60
    print("Time min.", min)


if __name__ == "__main__":
    main(parse_args())
