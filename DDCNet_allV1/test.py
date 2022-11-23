import argparse
import os,sys
import numpy as np
import torch
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader1
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils1.loss import SegmentationLosses
from utils1.calculate_weights import calculate_weigths_labels
from utils1.lr_scheduler import LR_Scheduler
from utils1.saver import Saver
from utils1.summaries import TensorboardSummary
from utils1.metrics import Evaluator
class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        
        self.summary2 = TensorboardSummary(self.saver.experiment_dir+'/test')
        self.writer1 = self.summary2.create_summary()
        
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.test_loader,self.nclass = make_data_loader1(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
       
        self.model  = model
        if os.path.exists('checkpoint.pth.tar'):
            model_dict = torch.load('checkpoint.pth.tar')
            self.model.load_state_dict(model_dict['state_dict'])
            print('load pretrained model sucessfully.')
        
    
    def test(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        flag1 = False
        correct = list(0. for i in range(self.nclass))
        total = list(0. for i in range(self.nclass))
        acc = list(0. for i in range(self.nclass))
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            # if self.args.cuda:
                # image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            global_step = i
            self.summary2.visualize_image(self.writer1, self.args.dataset, image, output, output, global_step)
        
            

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--epochs_p', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
        
    trainer = Trainer(args)
    trainer.test()
    
    trainer.writer1.close()

if __name__ == "__main__":
   main()
