import datetime
import os
import time
import sys

import numpy as np
import torch
import torchvision
from torch import nn
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler, UniformClipSampler

import data
from data.kinetics import Kinetics400
from data.video import VideoList
from model import CRW
import utils


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, print_freq,
    vis=None, checkpoint_fn=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)

    for step, (video, orig) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()

        video = video.to(device)
        output, loss, diagnostics = model(video)
        loss = loss.mean()

        if vis is not None and np.random.random() < 0.01:
            vis.wandb_init(model)
            vis.log(dict(loss=loss.mean().item()))
            vis.log({k: v.mean().item() for k,v in diagnostics.items()})

        if checkpoint_fn is not None and np.random.random() < 0.005:
            checkpoint_fn()
 
        optimizer.zero_grad()
        loss.backward()
        # print(torch.nn.utils.clip_grad_norm_(model.parameters(), 1), 'grad norm')
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(video.shape[0] / (time.time() - start_time))
        lr_scheduler.step()

    checkpoint_fn()

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def collate_fn(batch):
    # remove audio from the batch
    batch = [d[0] for d in batch]
    return default_collate(batch)

def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print("Preparing training dataloader")
    traindir = os.path.join(args.data_path, 'train_256' if not args.fast_test else 'val_256')
    valdir = os.path.join(args.data_path, 'val_256')

    st = time.time()
    cache_path = _get_cache_path(traindir)

    transform_train = utils.augs.get_train_transforms(args)

    def make_dataset(is_train, cached=None):
        _transform = transform_train if is_train else transform_test

        if 'kinetics' in args.data_path.lower():
            return Kinetics400(
                traindir if is_train else valdir,
                frames_per_clip=args.clip_len,
                step_between_clips=1,
                transform=transform_train,
                extensions=('mp4'),
                frame_rate=args.frame_skip,
                # cached=cached,
                _precomputed_metadata=cached
            )
        elif os.path.isdir(args.data_path): # HACK assume image dataset if data path is a directory
            return torchvision.datasets.ImageFolder(
                root=args.data_path,
                transform=_transform)
        else:
            return VideoList(
                filelist=args.data_path,
                clip_len=args.clip_len,
                is_train=is_train,
                frame_gap=args.frame_skip,
                transform=_transform,
                random_clip=True,
            )

    if args.cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
        cached = dict(video_paths=dataset.video_clips.video_paths,
                video_fps=dataset.video_clips.video_fps,
                video_pts=dataset.video_clips.video_pts)
        dataset = make_dataset(is_train=True, cached=cached)
        dataset.transform = transform_train
    else:
        dataset = make_dataset(is_train=True)
        if args.cache_dataset and 'kinetics' in args.data_path.lower():
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            dataset.transform = None
            torch.save((dataset, traindir), cache_path)
    
    if hasattr(dataset, 'video_clips'):
        dataset.video_clips.compute_clips(args.clip_len, 1, frame_rate=args.frame_skip)
        
    print("Took", time.time() - st)

    def make_data_sampler(is_train, dataset):
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = RandomClipSampler #UniformClipSampler
            return _sampler(dataset.video_clips, args.clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None
    
    print("Creating data loaders")
    train_sampler = make_data_sampler(True, dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, # shuffle=not args.fast_test,
        sampler=train_sampler, num_workers=args.workers//2,
        pin_memory=True, collate_fn=collate_fn)
    
    vis = utils.visualize.Visualize(args) if args.visualize else None

    print("Creating model")
    model = CRW(args, vis=vis).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.data_parallel:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module
    
    if args.partial_reload:
        checkpoint = torch.load(args.partial_reload, map_location='cpu')
        utils.partial_load(checkpoint['model'], model_without_ddp)        
        optimizer.param_groups[0]["lr"] = args.lr
        args.start_epoch = checkpoint['epoch'] + 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    def save_model_checkpoint():
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq,
                        vis=vis, checkpoint_fn=save_model_checkpoint)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = utils.arguments.train_args()
    main(args)
