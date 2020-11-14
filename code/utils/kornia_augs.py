import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import kornia
import kornia.augmentation as K

import numpy as np
import augs

IMG_MEAN = augs.IMG_MEAN
IMG_STD  = augs.IMG_STD

'''
Seems to be slower than the torchvision augs,
since we use cpu for kornia in the dataloading threads.
'''

class MapTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vid):
        return torch.stack([self.transforms(v) for v in vid])
    

def patch_grid(x, transform, shape=(64, 64, 3), stride=[1.0, 1.0]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]

    def transform(x):
        spatial_jitter = K.RandomResizedCrop(size=shape[:2], scale=(0.7, 0.9), ratio=(0.7, 1.3))

        import time
        t0 = time.time()
        x1 = x.unfold(2, 64, 32).unfold(3, 64, 32)
        t1 = time.time()
        x = kornia.contrib.extract_tensor_patches(x, 
            window_size=shape[:2], stride=stride[:2])
        t2 = time.time()
        print(t2-t1, t1-t0)

        T, N, C = x.shape[:3]
        x = transform(spatial_jitter(x.flatten(0,1))).view(T, N*C, *x.shape[3:])

        return x

    return transform

    
def get_frame_aug(frame_aug, patch_size):
    tt = []

    if 'cj' in frame_aug:
        _cj = 0.1
        tt += [
            #K.RandomGrayscale(p=0.2),
            K.ColorJitter(_cj, _cj, _cj, 0),
        ]

    if 'flip' in frame_aug:
        tt += [
            K.RandomHorizontalFlip(same_on_batch=True),
        ]

    tt += [kornia.color.Normalize(mean=IMG_MEAN, std=IMG_STD)]
    transform = nn.Sequential(*tt)
    print('Frame augs:', transform, frame_aug)

    if 'grid' in frame_aug:
        aug = patch_grid(x, transform=transform,
            shape=patch_size, stride=patch_size // 2)
    else:
        aug = transform
    
    return aug



def get_frame_transform(frame_transform_str, img_size, cuda=True):
    tt = []

    if 'gray' in frame_transform_str:
        tt += [K.RandomGrayscale(p=1)]

    if 'crop' in frame_transform_str:
        tt += [K.RandomResizedCrop(img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3))]
    else:
        tt += [kornia.geometry.transform.Resize((img_size, img_size))]

    if 'cj' in frame_transform_str:
        _cj = 0.1
        tt += [#K.RandomGrayscale(p=0.2), 
                K.ColorJitter(_cj, _cj, _cj, _cj)]

    if 'flip' in frame_transform_str:
        tt += [K.RandomHorizontalFlip()]

    return tt


def get_train_transform(args, cuda=True):
    imsz = args.img_size
    norm_size = kornia.geometry.transform.Resize((imsz, imsz))
    norm_imgs = kornia.color.Normalize(mean=IMG_MEAN, std=IMG_STD)

    frame_transform = get_frame_transform(args.frame_transform, imsz, cuda)
    frame_aug = get_frame_aug(args.frame_aug, args.patch_size)

    transform = transforms.Compose(frame_transform + frame_aug)
    plain = nn.Sequential(norm_size, norm_imgs)

    def with_orig(x):
        if cuda:
            x = x.cuda()

        if x.max() > 1 and x.min() >= 0:
            x = x.float()
            x -= x.min()
            x /= x.max()

        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        x = (transform(x) if patchify else plain(x)).cpu(), \
                plain(x[0:1]).cpu()

        return x

    return with_orig
