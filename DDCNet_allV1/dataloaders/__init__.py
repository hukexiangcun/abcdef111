from dataloaders.datasets import pascal#cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_p_set = pascal.VOCSegmentation(args, split='train_p')
        #train_n_set = pascal.VOCSegmentation(args, split='train_n')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_p_set.NUM_CLASSES
        train_p_loader = DataLoader(train_p_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        #train_n_loader = DataLoader(train_n_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        
        return train_p_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
def make_data_loader1(args, **kwargs):

    if args.dataset == 'pascal':
        test_set = pascal.VOCSegmentation(args, split='test')
        #train_n_set = pascal.VOCSegmentation(args, split='train_n')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        return test_loader, num_class

