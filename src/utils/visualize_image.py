# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import numpy as np
import torch
from PIL import Image

def save_visualization(by_full, out, vis_path, suffix):
    vgg_std = (0.229, 0.224, 0.225)
    vgg_mu = (0.485, 0.456, 0.406)
    post_proc = []
    for arrays_list in (by_full,out):
        arrays_list = arrays_list.detach().numpy()
        N, C, W, H = arrays_list.shape
        # de-normalize the transofrmed data: (y*s + m)*255
        # make a standard deviation array
        std_ = np.array(vgg_std).repeat((N*W*H)).reshape((C,N,W,H)).transpose(1,0,2,3)
        # make a standard deviation array
        mu_ = np.array(vgg_mu).repeat((N*W*H)).reshape((C,N,W,H)).transpose(1,0,2,3)
        # backtransform from vgg normalization
        arrays_list = ((arrays_list*std_+mu_)*255).clip(0,255).astype(np.uint8)
        # append to post-processed
        post_proc.append(arrays_list)
    
    # concatenate
    img_cat = np.concatenate(post_proc, axis = 2)
    for indexImage in range(N):
        # convert to image
        img = np.array(img_cat[indexImage])
        img = Image.fromarray(np.transpose(img, (1,2,0)))
        # save
        path_to_saved_image = os.path.join(vis_path,"eval_%s_%d.jpg" % (suffix, indexImage))
        img.save(path_to_saved_image, "JPEG", quality=80, optimize=True, progressive=True)
