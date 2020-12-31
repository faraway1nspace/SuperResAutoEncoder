import sys
import time
import os
import json
import pickle as pkl

import torch
import torchvision.transforms as Transforms
import albumentations as A
import cv2
import random

from utils.utils import rebase_path
from utils.visualize_image import save_visualization
from utils.image_transform import pil_loader, ToTensorV2
from attrib_dataset import AttribDataset

import importlib
import json

import numpy as np

import matplotlib.pyplot as plt

from nets.transformer_net import *

from nets.vgg import Vgg16

mse_loss = torch.nn.MSELoss()

# make the image data loader
class DataLoaderTransformer:
    def __init__(self,
                 path_db, # path to database
                 albumentations_transformations=None,
                 selectedAttributes = None, # select according to certain attributes
                 size_compressed_input = None
                 ):
        # a wrapper class that invokes an Albumentations Compose, conditional on a 'size' argument
        self.albumentations_transformations = albumentations_transformations
        
        # path to dataset, separated into different class-directories
        self.path_db = path_db
        
        self.selectedAttributes = selectedAttributes
        
        # build key order
        self.attribKeysOrder = self.getDataset(size=10).getKeyOrders()
        print("AC;classes : ")
        print(self.attribKeysOrder)
        
        #
        if size_compressed_input is None:
            size_compressed_input = 64
            print("setting 'input' image size to %d" % size_compressed_input)
        self.size_compressed_input = size_compressed_input
    
    def getDBLoader(self, size, bs, num_workers = None, shuffle=None):
        r"""
        Load the training dataset for the given scale.
        Args:
            - scale (int): scale at which we are working
        Returns:
            A dataset with properly resized inputs.
        """
        if shuffle is None:
            shuffle =True
        
        if num_workers is None:
            num_workers = 1
        
        dataset = self.getDataset(size)
        
        return torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers) 
    
    def getDataset(self, size=None):
        
        if size is None:
            size = 512
        
        print("size", size)
        if self.albumentations_transformations is None:
            raise NotImplementedError("need to provide a class of Albumentations Compose called argument 'albumentations_transformations'")
        
        transform = self.albumentations_transformations(size)
        
        return AttribDataset(self.path_db,
                             transform=transform,
                             attribDictPath=None, #self.pathAttribDict,
                             specificAttrib=self.selectedAttributes,
                             mimicImageFolder=True)#self.imagefolderDataset)
    
    def downscale_fullsize_target_to_small_input(self, target_images, size_compressed_input=None):
        """ take the target image (generally 512x512) and downscale it to 64x64"""
        if size_compressed_input is None:
            size_compressed_input = self.size_compressed_input
            if size_compressed_input is None:
                raise AttributeError("'size_compressed_input' is None; must set to ~64")
        # change to Numpy
        #target_image = target_image.detach().numpy()
        return Transforms.Resize(size_compressed_input, interpolation=2)(target_images)

class AlbumentationsTransformations(object):
    """ Wrapper class used to re-initialize a Albumentations Compose, conditional on changing parameters of 'size' """
    def __init__(self, size):
        if isinstance(size,int):
            size = (size,size)
        self.A_Compose = A.Compose([A.ChannelShuffle(always_apply=False, p=0.1),
                                    A.RandomGridShuffle(always_apply=False, p=0.05, grid=(2, 2)),
                       A.transforms.GridDistortion(num_steps=5, distort_limit=0.4, p=0.6),
                       A.HorizontalFlip(p=0.5),
                       A.RandomResizedCrop(always_apply=True, p=1.0, height=size[0], width=size[0],
                                         scale=(0.75, 1),
                                         ratio=(0.97, 1.03), interpolation=3),#ONLY 3 SEEMS TO WORK
                       A.transforms.ColorJitter(brightness=0.05, contrast=0.07, saturation=0.04, hue=0.08, always_apply=False, p=0.7),#HSV random
                       A.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
                       ToTensorV2()
                       ])
    
    def __call__(self, PIL_image):
        transformed_image_dict = self.A_Compose(image = np.array(PIL_image))
        return transformed_image_dict['image']
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

def gram_matrix(y):
    """ gram matrix for style loss"""
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def style_lossf(gram_style_fullsize, gram_style_reconstruction):
    style_loss = 0.
    # loop through vgg features
    for gm_o, gm_r in zip(gram_style_fullsize, gram_style_reconstruction):
        style_loss += mse_loss(gm_o, gm_r)
    
    return style_loss 

def train(params_dict,
          path_db,
          save_dir,
          model_type = None, # either "TransformerNet4" or "NetResidUpsample"
          do_cuda = None,
          do_reload=None,
          n_epochs = None,
          bs = None,
          lr = None,
          wts = None,
          path_to_reload_model = None,
          sleep=None,
          import_dir = None,
          save_model_every_X_iterations = None):
    
    if model_type is None:
        model_type = "TransformerNet4"
    
    if do_cuda is None:
        do_cuda = False
    
    if n_epochs is None:
        n_epochs = 10
    
    if bs is None:
        bs = 3
    
    if lr is None:
        lr = 0.001
    
    if wts is None:
        wts = {'style':1e10, 'pixel':512.0**-0.5, 'content':1e5}
    
    if save_model_every_X_iterations is None:
        save_model_every_X_iterations = 100
    
    mse_loss = torch.nn.MSELoss()
    
    size_compressed_input = (64,64)
    
    # initialize the db loader class
    db_loader = DataLoaderTransformer(
        path_db = path_db, #'/tmp/data/', 
        albumentations_transformations=AlbumentationsTransformations,
        selectedAttributes=None,
        size_compressed_input = size_compressed_input)
    
    params = HyperParams(params_dict)
    
    device = torch.device("cuda" if do_cuda else "cpu")
        
    #initialize the VGG pre-trained model for the perceptual losses
    vgg = Vgg16(requires_grad=False).to(device)
    
    # initialize the model
    if model_type == "TransformerNet4":
        net2 = TransformerNet4(params).to(device)
    elif model_type == "NetResidUpsample":
        net2 = NetResidUpsample(params).to(device)
    else:
        net2 = TransformerNet4(params).to(device)
    
    optimizer = torch.optim.Adam(net2.parameters(),lr = lr)
    
    #save_dir = "models_trained/v2/";
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    target_size = 512
        
    epoch = -1
    
    if do_reload is None:
        do_reload = False
    
    if do_reload:
        # check if there is a saved json file with the path to the latest model
        path_to_lastest_model_json = os.path.join(save_dir, "latest_model.json")
        if os.path.isfile(path_to_lastest_model_json):
            with open(path_to_lastest_model_json,'r') as jcon:
                path_to_reload_model_from_json = json.load(jcon)['path_to_latest_model']
            
            if (not path_to_reload_model is None):
                if (path_to_reload_model_from_json != path_to_reload_model):
                    print("The reload model (%s) is different from the latest model %s; defaulting to the latest saved" % (path_to_reload_model,path_to_reload_model_from_json))
            
            path_to_reload_model = path_to_reload_model_from_json
        
        if os.path.isfile(path_to_reload_model):
            print("reloading model %s" % path_to_reload_model)
            previous_model_state = torch.load(path_to_reload_model)
            epoch = previous_model_state['epoch']
            print("resuming at epoch %d" % epoch)
            net2.load_state_dict(previous_model_state['model_state_dict'])
            optimizer = torch.optim.Adam(net2.parameters(),lr = lr)
            optimizer.load_state_dict(previous_model_state['optimizer_state_dict'])
        else:
            print("STARTING FRESH MODEL AT 0")
    
    while epoch < n_epochs:
        epoch+=1
        db_iter = db_loader.getDBLoader(size=target_size, bs=bs, num_workers = 1, shuffle=True)
        # loop through the images
        for bitem, (by_full, blabels) in enumerate(db_iter):
            
            # compress the image to 64x64  (serves as the input image
            bx_comp = db_loader.downscale_fullsize_target_to_small_input(target_images = by_full)
            if do_cuda:
                bx_comp = bx_comp.to(device)
                by_full = by_full.to(device)
            
            optimizer.zero_grad()
            # reconstruct image
            out, out_downsampled = net2(bx_comp)
            # VGG features
            features_style_fullsize = vgg(by_full)
            features_style_reconstruction = vgg(out)
            
            # gram matrices for style loss
            gram_style_fullsize = [gram_matrix(y) for y in features_style_fullsize]
            # style of the reconstruction
            gram_style_reconstruction = [gram_matrix(y) for y in features_style_reconstruction]
            # style loss
            bstyle_loss = style_lossf(gram_style_fullsize, gram_style_reconstruction)
            
            # pixel loss
            pixel_loss = mse_loss(by_full, out)
            # content loss (vgg deeper features
            #content_loss = mse_loss(bx_comp, out_downsampled)
            perceptual_loss = mse_loss(features_style_fullsize.relu2_2, features_style_reconstruction.relu2_2) # 
            
            #bstyle_loss.backward() # takes a long time
            rwts_pixel = wts['pixel']*(random.random()<0.95);
            #rwts_content = wts['content']*(random.random()<0.95)
            (rwts_pixel*pixel_loss + wts['style']*bstyle_loss + wts['content']*perceptual_loss).backward()
            optimizer.step()
            
            if bitem % save_model_every_X_iterations == 0:
                print("saving model and making a dummy image")
                path_to_model = "%sstyletransfer_v1_e%d.model" % (save_dir, epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net2.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'style_loss':bstyle_loss.item(),
                    'content_loss':perceptual_loss.item(),
                    'pixel_loss':pixel_loss.item(),
                    'lr':lr,
                    'weights':wts,
                    'bs':bs,
                    'params_dict':params_dict},path_to_model)
                
                # save the path to the latest model
                path_to_lastest_model_json = os.path.join(os.path.split(path_to_model)[0],"latest_model.json")
                with open(path_to_lastest_model_json,'w') as jcon:
                    json.dump({'path_to_latest_model':path_to_model}, jcon)
                
                # plot images
                save_visualization(by_full, out, vis_path=save_dir, suffix = "E%d" % epoch)
            
            if bitem % 10 ==0:
                print("E%d; STEP%d; style:%0.6f; content:%0.3f; pix:%0.3f" % (epoch, bitem, bstyle_loss.item(), perceptual_loss.item(), pixel_loss.item()))
                if not sleep is None:
                    time.sleep(sleep)
        
        if True:
            print("saving model and making a dummy image")
            path_to_model = "%sstyletransfer_v1_e%d.model" % (save_dir, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'style_loss':bstyle_loss.item(),
                'content_loss':perceptual_loss.item(),
                'pixel_loss':pixel_loss.item(),
                'lr':lr,
                'weights':wts,
                'bs':bs,
                'params_dict':params_dict},path_to_model)
            
            # save the path to the latest model
            path_to_lastest_model_json = os.path.join(os.path.split(path_to_model)[0],"latest_model.json")
            with open(path_to_lastest_model_json,'w') as jcon:
                json.dump({'path_to_latest_model':path_to_model}, jcon)
            
            # plot images
            save_visualization(by_full, out, vis_path=save_dir, suffix = "E%d" % epoch)
            if not sleep is None:
                time.sleep(5*sleep)

