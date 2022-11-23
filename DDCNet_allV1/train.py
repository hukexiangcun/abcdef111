import argparse
import os,sys
import numpy as np
import torch
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
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
        self.summary1 = TensorboardSummary(self.saver.experiment_dir+'/train')
        self.summary2 = TensorboardSummary(self.saver.experiment_dir+'/val')
        self.writer = self.summary1.create_summary()
        self.writer1 = self.summary2.create_summary()
        
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_p_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        model_p = Conv_P(args.batch_size)
        model_n = Conv_N()
        
        
        train_params_p = [{'params': model_p.get_1x_lr_params_p(), 'lr': args.lr},
                         {'params': model.get_1x_lr_params(), 'lr': args.lr},
                         {'params': model.get_10x_lr_params(), 'lr': 10*args.lr}
                        ]
                      
                        

        # Define Optimizer
        optimizer_p = torch.optim.SGD(train_params_p, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight_p = calculate_weigths_labels(args.dataset, self.train_p_loader, self.nclass)
                
            weight_p = torch.from_numpy(weight_p.astype(np.float32))
            
        else:
            weight = None
        self.criterion_p = SegmentationLosses(weight=weight_p, cuda=args.cuda).build_loss(mode=args.loss_type)
       
        self.model, self.model_p,  self.optimizer_p,  = model, model_p, optimizer_p
        if os.path.exists('checkpoint.pth.tar'):
            model_dict = torch.load('checkpoint.pth.tar')
            self.model.load_state_dict(model_dict['state_dict'])
            #self.optimizer_p.load_state_dict(model_dict['optimizer'])
            print('load pretrained model sucessfully.')
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler_p = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs_p, len(self.train_p_loader))
        
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            
            self.model_p = torch.nn.DataParallel(self.model_p, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model_p)
            self.model_p = self.model_p.cuda()
            
            
        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_save = 0.0
        # if args.resume is not None:
            # if not os.path.isfile(args.resume):
                # raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            # checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # if args.cuda:
                # self.model.module.load_state_dict(checkpoint['state_dict'])
            # else:
                # self.model.load_state_dict(checkpoint['state_dict'])
            # if not args.ft:
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.best_pred = checkpoint['best_pred']
            # print("=> loaded checkpoint '{}' (epoch {})"
                  # .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        self.best_acc = 0
    

    def training_p(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_p_loader)
        num_img_tr = len(self.train_p_loader)
        flag1 = False
        
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            
            
            self.scheduler_p(self.optimizer_p, i, epoch, self.best_pred)
            self.optimizer_p.zero_grad()
            x = self.model_p(image)
            output = self.model(x)
            loss = self.criterion_p(output, target)
            loss.backward()
            
            self.optimizer_p.step()
            train_loss = train_loss+loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            #if i % (num_img_tr // 1) == 0:
            global_step = i #+ num_img_tr * epoch
            
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        flag1 = False
        correct = list(0. for i in range(self.nclass))
        total = list(0. for i in range(self.nclass))
        acc = list(0. for i in range(self.nclass))
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                x = self.model_p(image)
                output = self.model(x)
            target1 = target
            loss = self.criterion_p(output, target)
            test_loss = test_loss+loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator

            global_step = i
            pred1 = pred
            target2 = target
            pred1 = pred1.flatten()
            target2 = target2.flatten()
            res = pred1 == target2
            for label_idx in range(len(target2)):
                label_single = target2[int(label_idx)]
                label_single = label_single.astype(np.int8)
                
                correct[label_single] = correct[label_single]+res[int(label_idx)].item()
                total[label_single] = total[label_single]+1
            if i == 1 or i == 3 or i == 7 or i == 9 or i == 11 or i == 18 or i == 25 or i == 35 or i == 41:
                self.summary2.visualize_image(self.writer1, self.args.dataset, image, target1, output, global_step)
            self.evaluator.add_batch(target, pred)
            
            
        # Fast test during the training
        for acc_idx in range(self.nclass):
           acc[int(acc_idx)] = correct[int(acc_idx)]/total[int(acc_idx)]
        Acc = self.evaluator.Pixel_Accuracy()
        
            
            
        # if Acc > self.best_acc:
            # state = {'net':self.model.state_dict()}
            # filepath = os.path.join('./', 'checkpoint_model_epoch.pth.tar') #最终参数模型
            # torch.save(state, filepath)
            # self.best_acc = Acc
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer1.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer1.add_scalar('val/mIoU', mIoU, epoch)
        self.writer1.add_scalar('val/Acc', Acc, epoch)
        self.writer1.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer1.add_scalar('val/fwIoU', FWIoU, epoch)
        
        
        
        print('Validation:')
        
        print('Acc for building',acc[1])
        print('Acc for ground',acc[2])
        print('Acc for tree',acc[3])
        print('Acc for car',acc[4])
        #print('Acc for person',acc[5])
        print('Acc for others',acc[0])
        
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = Acc
        if new_pred > self.best_pred:
           is_best = True
           self.best_pred = new_pred
           self.saver.save_checkpoint({
               'epoch': epoch + 1,
               'state_dict': self.model.module.state_dict(),
               'optimizer': self.optimizer_p.state_dict(),
               'best_pred': self.best_pred,
           }, is_best)

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
    #print(args.cuda)
    #sys.exit()
    #args.cuda =True
    flag = True
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 100,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs_p):
        trainer.training_p(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            
            trainer.validation(epoch)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        
        # trainer.training_n(epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            
            # trainer.validation(epoch)
    
    trainer.writer.close()

if __name__ == "__main__":
   main()
