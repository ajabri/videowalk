# Video's features
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io

import visdom
import time
import PIL
import torchvision
import torch

import matplotlib.pyplot as plt
from matplotlib import cm


def pca_feats(ff, K=1, solver='auto', whiten=True, img_normalize=True):
    ## expect ff to be   N x C x H x W

    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3*K,
        svd_solver=solver,
        whiten=whiten
    )

    ff = ff.transpose(1, 2).transpose(2, 3)
    ff = ff.reshape(N*H*W, C).numpy()
    
    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3*K)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    pca_ff = [pca_ff[:, kk:kk+3] for kk in range(0, pca_ff.shape[1], 3)]

    if img_normalize:
        pca_ff = [(x - x.min()) / (x.max() - x.min()) for x in pca_ff]

    return pca_ff[0] if K == 1 else pca_ff


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        video = (video*255).astype(np.uint8)
        
    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration = 0.2)


def draw_matches(x1, x2, i1, i2):
    # x1, x2 = f1, f2/
    detach = lambda x: x.detach().cpu().numpy().transpose(1,2,0) * 255
    i1, i2 = detach(i1), detach(i2)
    i1, i2 = cv2.resize(i1, (400, 400)), cv2.resize(i2, (400, 400))

    for check in [True]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=check)
        # matches = bf.match(x1.permute(0,2,1).view(-1, 128).cpu().detach().numpy(), x2.permute(0,2,1).view(-1, 128).cpu().detach().numpy())

        h = int(x1.shape[-1]**0.5)
        matches = bf.match(x1.t().cpu().detach().numpy(), x2.t().cpu().detach().numpy())

        scale = i1.shape[-2] / h
        grid = torch.stack([torch.arange(0, h)[None].repeat(h, 1), torch.arange(0, h)[:, None].repeat(1, h)])
        
        grid = grid.view(2, -1)
        grid = grid * scale + scale//2

        kps = [cv2.KeyPoint(grid[0][i], grid[1][i], 1) for i in range(grid.shape[-1])]

        matches = sorted(matches, key = lambda x:x.distance)

        # img1 = img2 = np.zeros((40, 40, 3))
        out = cv2.drawMatches(i1.astype(np.uint8), kps, i2.astype(np.uint8), kps,matches[:], None, flags=2).transpose(2,0,1)

    return out


import wandb
class Visualize(object):
    def __init__(self, args):

        self._env_name = args.name
        self.vis = visdom.Visdom(
            port=args.port,
            server='http://%s' % args.server,
            env=self._env_name,
        )
        self.args = args

        self._init = False

    def wandb_init(self, model):
        if not self._init:
            self._init = True
            wandb.init(project="videowalk", group="release", config=self.args)
            wandb.watch(model)

    def log(self, key_vals):
        return wandb.log(key_vals)

    def nn_patches(self, P, A_k, prefix='', N=10, K=20):
        nn_patches(self.vis, P, A_k, prefix, N, K)

    def save(self):
        self.vis.save([self._env_name])

def get_stride(im_sz, p_sz, res):
    stride = (im_sz - p_sz)//(res-1)
    return stride

def nn_patches(vis, P, A_k, prefix='', N=10, K=20):
    # produces nearest neighbor visualization of N patches given an affinity matrix with K channels

    P = P.cpu().detach().numpy()
    P -= P.min()
    P /= P.max()

    A_k = A_k.cpu().detach().numpy() #.transpose(-1,-2).numpy()
    # assert np.allclose(A_k.sum(-1), 1)

    A = np.sort(A_k, axis=-1)
    I = np.argsort(-A_k, axis=-1)
    
    vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header' %(prefix))

    for n,i in enumerate(np.random.permutation(P.shape[0])[:N]):
        p = P[i]
        vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header_%s' % (prefix, n))
        # vis.image(p,  win='%s_patch_query_%s' % (prefix, n))

        for k in range(I.shape[0]):
            vis.images(P[I[k, i, :K]], nrow=min(I.shape[-1], 20), win='%s_patch_values_%s_%s' % (prefix, n, k))
            vis.bar(A[k, i][::-1][:K], opts=dict(height=150, width=500), win='%s_patch_affinity_%s_%s' % (prefix, n, k))


def compute_flow(corr):
    # assume batched affinity, shape N x H * W x W x H
    h = w = int(corr.shape[-1] ** 0.5)

    # x1 -> x2
    corr = corr.transpose(-1, -2).view(*corr.shape[:-1], h, w)
    nnf = corr.argmax(dim=1)

    u = nnf % w # nnf.shape[-1]
    v = nnf / h # nnf.shape[-2] # nnf is an IntTensor so rounds automatically

    rr = torch.arange(u.shape[-1])[None].long().cuda()

    for i in range(u.shape[-1]):
        u[:, i] -= rr

    for i in range(v.shape[-1]):
        v[:, :, i] -= rr

    return u, v

def vis_flow_plt(u, v, x1, x2, A):
    flows = torch.stack([u, v], dim=-1).cpu().numpy()
    I, flows = x1.cpu().numpy(), flows[0]

    H, W = flows.shape[:2]
    Ih, Iw, = I.shape[-2:]
    mx, my = np.mgrid[0:Ih:Ih/(H+1), 0:Iw:Iw/(W+1)][:, 1:, 1:]
    skip = (slice(None, None, 1), slice(None, None, 1))

    ii = 0
    fig, ax = plt.subplots()
    im = ax.imshow((I.transpose(1,2,0)),)
    
    C = cm.jet(torch.nn.functional.softmax((A * A.log()).sum(-1).cpu(), dim=-1))
    ax.quiver(my[skip], mx[skip], flows[...,0][skip], flows[...,1][skip]*-1, C)#, scale=1, scale_units='dots')
    # ax.quiver(mx[skip], my[skip], flows[...,0][skip], flows[...,1][skip])

    return plt
    
def frame_pair(x, ff, mm, t1, t2, A, AA, xent_loss, viz):
    normalize = lambda xx: (xx-xx.min()) / (xx-xx.min()).max()
    spatialize = lambda xx: xx.view(*xx.shape[:-1], int(xx.shape[-1]**0.5), int(xx.shape[-1]**0.5))

    N = AA.shape[-1]
    H = W = int(N**0.5)
    AA = AA.view(-1, H * W, H, W)

    ##############################################
    ## Visualize PCA of Embeddings, Correspondences
    ##############################################

    # import pdb; pdb.set_trace()
    if (len(x.shape) == 6 and x.shape[1] == 1):
        x = x.squeeze(1)

    if len(x.shape) < 6:   # Single image input, no patches
        # X here is B x C x T x H x W
        x1, x2 = normalize(x[0, :, t1]), normalize(x[0, :, t2])
        f1, f2 = ff[0, :, t1], ff[0, :, t2]
        ff1 , ff2 = spatialize(f1), spatialize(f2)

        xx = torch.stack([x1, x2]).detach().cpu()
        viz.images(xx, win='imgs')

        # Flow
        u, v = compute_flow(A[0:1])
        flow_plt = vis_flow_plt(u, v, x1, x2, A[0])
        viz.matplot(flow_plt, win='flow_quiver')

        # Keypoint Correspondences
        kp_corr = draw_matches(f1, f2, x1, x2)
        viz.image(kp_corr, win='kpcorr')

        # # PCA VIZ
        pca_ff = pca_feats(torch.stack([ff1,ff2]).detach().cpu())
        pca_ff = make_gif(pca_ff, outname=None)
        viz.images(pca_ff.transpose(0, -1, 1, 2), win='pcafeats', opts=dict(title=f"{t1} {t2}"))

    else:  # Patches as input
        # X here is B x N x C x T x H x W
        x1, x2 =  x[0, :, :, t1],  x[0, :, :, t2]
        m1, m2 = mm[0, :, :, t1], mm[0, :, :, t2]

        pca_ff = pca_feats(torch.cat([m1, m2]).detach().cpu())
        pca_ff = make_gif(pca_ff, outname=None, sz=64).transpose(0, -1, 1, 2)
        
        pca1 = torchvision.utils.make_grid(torch.Tensor(pca_ff[:N]), nrow=int(N**0.5), padding=1, pad_value=1)
        pca2 = torchvision.utils.make_grid(torch.Tensor(pca_ff[N:]), nrow=int(N**0.5), padding=1, pad_value=1)
        img1 = torchvision.utils.make_grid(normalize(x1)*255, nrow=int(N**0.5), padding=1, pad_value=1)
        img2 = torchvision.utils.make_grid(normalize(x2)*255, nrow=int(N**0.5), padding=1, pad_value=1)
        viz.images(torch.stack([pca1,pca2]), nrow=4, win='pca_viz_combined1')
        viz.images(torch.stack([img1.cpu(),img2.cpu()]), opts=dict(title=f"{t1} {t2}"), nrow=4, win='pca_viz_combined2')
    
    ##############################################
    # LOSS VIS
    ##############################################
    color = cm.get_cmap('winter')

    xx = normalize(xent_loss[:H*W])
    img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] 
                for aa in AA[0, :, :, :, None].cpu().detach().numpy()]
    img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
    img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
    img_grid = torch.from_numpy(np.array(img_grid))
    img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
    
    # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    viz.images(img_grid, win='lossvis')

