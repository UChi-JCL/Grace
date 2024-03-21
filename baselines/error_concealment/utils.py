import matplotlib.patches as patches
from matplotlib.path import Path
import os
import sys
import io
import cv2
import matplotlib.pyplot as plt
import time
import argparse
import shutil
import random
import zipfile
from glob import glob
import math
import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFilter
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.distributions import Uniform

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')



# #####################################################
# #####################################################

class ZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = ZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, image_name):
        zfile = ZipReader.build_file_dict(path)
        data = zfile.read(image_name)
        im = Image.open(io.BytesIO(data))
        return im

# ###########################################################################
# ###########################################################################


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################



# ##############################################
# ##############################################


def save_img(img, path, denorm=False):
    B, C,H,W = img.size()
    if denorm:
        img = (img+1)/2.
    if C is not 3:
        img = img.repeat(1,3,1,1)
    img = img[0]
    img = img.permute(1,2,0)
    img = img.cpu().data.numpy()
    img = (np.clip(img, 0,1)*255).astype(np.uint8)
    Image.fromarray(img).save(path)

def save_img_np(img, path):
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def psnr_measure(src ,tar, shave_border=0):

    def psnr(y_true,y_pred, shave_border=4):
        '''
            Input must be 0-255, 2D
        '''

        target_data = np.array(y_true, dtype=np.float32)
        ref_data = np.array(y_pred, dtype=np.float32)

        diff = ref_data - target_data
        if shave_border > 0:
            diff = diff[shave_border:-shave_border, shave_border:-shave_border]
        rmse = np.sqrt(np.mean(np.power(diff, 2)))

        # print('rmse', rmse)
        # if rmse < 1:
        #     return 50
        # else:
        return 20 * np.log10(255./rmse)

    def rgb2ycbcr(img, maxVal=255):
        O = np.array([[16],
                    [128],
                    [128]])
        T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                    [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                    [0.439215686274510, -0.367788235294118, -0.071427450980392]])

        if maxVal == 1:
            O = O / 255.0

        img_s = img.shape
        if len(img_s) >= 3:
            t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
        else:
            t = img
        t = np.dot(t, np.transpose(T))
        t[:, 0] += O[0]
        t[:, 1] += O[1]
        t[:, 2] += O[2]
        if len(img_s) >= 3:
            ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
        else:
            ycbcr = t

        return ycbcr

    # return psnr(rgb2ycbcr((src).astype(np.uint8))[:,:,0], rgb2ycbcr((tar).astype(np.uint8))[:,:,0], shave_border=shave_border)
    return psnr(src, tar, shave_border=shave_border)



##########################################################
##########################################################

def tensor_to_numpy(x, denorm=True):
    if denorm:
        x = (x + 1)/2.
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    x= np.clip(x, 0,1)*255
    return x

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)




def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=1)
    
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=1)
    grid = grid.permute(0,2,3,1)
    
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    dim = len(x.size())
    if dim == 5:
        B,T,C,H,W = x.size()
        x = x.view(B*T, C,H,W)
        flo = flo.view(B*T, 2, H, W)
    else:
        T = 1 

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)

    mask = torch.autograd.Variable(torch.ones(x.size())) 
    if x.is_cuda:
        mask = mask.cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    output = output*mask
    if dim == 5:
        output = output.view(-1,T,C,H,W)

    return output


def trace_all_flows(flowFs, flowBs, src_idx=0, tar_idx=0):

    T,C,H,W = flowFs.size()
    flow = torch.zeros_like(flowFs[0:1]).cuda()
    pixel = coords_grid(1,H,W).cuda()

    if src_idx > tar_idx:
        reverse = True
    else:
        reverse = False

    # print(forward)
    if reverse:

        for idx in range(src_idx - 1, tar_idx-1, -1):
            disp = bilinear_sampler(flowBs[idx:idx+1].contiguous(), pixel)
            pixel = pixel + disp
            flow = flow + disp
        
    else:
        for idx in range(src_idx, tar_idx):
            disp = bilinear_sampler(flowFs[idx:idx+1].contiguous(), pixel)
            pixel = pixel + disp
            flow = flow + disp
        
            
    return flow

def trace_flows_from_idx(flowFs, flowBs, idx=0):
    T,C,H,W = flowFs.size()
    T = T+1
    zero_flow = torch.zeros_like(flowFs[0:1]).cuda()
    
    trace_flows = torch.zeros(T,C,H,W).cuda()
    cnt = 0 
    while cnt < T:
        if idx == cnt:
            trace_flows[idx:idx+1,:,:,:] = zero_flow
            cnt +=1
        elif idx < cnt:
            pixel = coords_grid(1,H,W).cuda()
            flowF =  zero_flow
            for f in range(idx+1, T):
                disp = bilinear_sampler(flowFs[f-1:f].contiguous(), pixel)
                pixel = pixel + disp
                flowF = flowF + disp
                trace_flows[f:f+1,:,:,:] = flowF
                cnt +=1
        else: # idx > eidx
            pixel = coords_grid(1,H,W).cuda()
            flowB = zero_flow
            for f in range(idx, 0, -1):
                disp = bilinear_sampler(flowBs[f-1:f].contiguous(), pixel)
                pixel = pixel + disp
                flowB = flowB + disp
                trace_flows[f-1:f,:,:,:] = flowB
                cnt +=1

    return trace_flows

        
def find_ref_indexs(tar_indexs=0, T=10, stride=1):

    ref_indexs = []
    ref_len = []

    for idx in tar_indexs:
        ref_index = []

        for i in range(idx+stride, T, stride):
            ref_index.append(i)
        if T-1 not in ref_index:
            ref_index.append(T-1)
        
        for i in range(idx-stride, 0, -1*stride):
            ref_index.append(i)
        if 0 not in ref_index:
            ref_index.append(0)

        tmp_index = ref_index
        tmp_index = [abs(i - idx) for i in tmp_index]
        _, ref_index = zip(*sorted(zip(tmp_index, ref_index)))

        ref_indexs.append(list(ref_index))
        ref_len.append(len(ref_index))

    return ref_indexs, min(ref_len)


# for finding nearst key frame.
def find_ref_key_indexs(tar_indexs, key_indexs):
    ref_indexs = []
    ref_len = []

    if len(key_indexs) == 0:
        return [tar_indexs.copy()], 0

    for idx in tar_indexs:
            
        tmp_index = key_indexs
        tmp_index = [abs(kidx - idx) for kidx in tmp_index]

        _, tmp_key_indexs = zip(*sorted(zip(tmp_index, key_indexs)))

        ref_indexs.append(list(tmp_key_indexs))
        ref_len.append(len(tmp_key_indexs))

    
    return ref_indexs, min(ref_len)



    
def find_key_index(masks, flowFs, flowBs, stride=3, key_indexs = None):

    done = False

    T,_,_,_ = masks.size()
    mask_sums = torch.zeros(T)
    tmp_masks = masks.clone()
    zero_mask = torch.zeros_like(masks[0:1]).cuda()
    
    # Iteratively find key index 
    # Temporarily fill masks with the guidance of compelted flows for finding the largest remainig hole
    for idx in range(T):

        if key_indexs is None:
            ref_indexs, _ = find_ref_indexs([idx], T=T, stride=stride)
            ref_indexs = ref_indexs[0]
        else:
            if idx in key_indexs:
                continue
            ref_indexs = key_indexs

        # Trace all flows from frame_{idx} to other frames
        trace_flows = trace_flows_from_idx(flowFs, flowBs, idx=idx)
        
        tar_mask = masks[idx:idx+1].clone()

        for l, ridx in enumerate(ref_indexs):

            ref_mask = masks[ridx:ridx+1]
            flow = trace_flows[ridx:ridx+1]

            # Fill the target mask
            warpM = 1 - dilation(1-warp(1-ref_mask, flow), iter=1)
            tar_mask = ((tar_mask - warpM)>0.5).float()

            if torch.sum(tar_mask) < 1:
                break
        # Save the results
        masks[idx:idx+1] = tar_mask.clone()

    

        
    if torch.sum(masks) > 1 :
        # If there are remaninig holes ... 
        if key_indexs is None:
            key_indexs = []

        # Find the largest remaining hole
        mask_sums = torch.sum(masks, axis=(1,2,3))
        kidx = torch.argmax(mask_sums)
        kidx = kidx.item()

        key_indexs.append(kidx)

        for kidx in key_indexs:
            tmp_masks[kidx:kidx+1] = zero_mask.clone()
        
    else:
        # If every holes are filled, set center frame as first key frame
        if key_indexs is None:   
            done = False
            first_key_idx = T//2
            key_indexs = [first_key_idx]
            tmp_masks[first_key_idx:first_key_idx+1] = zero_mask.clone()
        else:
            # If key indexs are already finded, then stop process
            done = True

    return key_indexs, tmp_masks, done
    

def find_key_indexs(masks, flowFs, flowBs, stride=3, key_stride= 15):
        
    T,_,_,_ = masks.size()
    tmp_masks = masks.clone()
    
    done = False
    key_indexs = None

    zero_mask = torch.zeros_like(masks[0:1]).cuda()

    cnt = 0
    while done is False:
        cnt = cnt + 1
        key_indexs, tmp_masks, done = find_key_index(tmp_masks, flowFs, flowBs, stride=stride, key_indexs=key_indexs)

        # After find the first key frame, add stride key index
        # This is mainly due to reduce computing time
        if cnt == 1:
            key_indexs = add_stride_key_index(T, key_indexs, key_stride=key_stride)
            for kidx in key_indexs:
                tmp_masks[kidx:kidx+1] = zero_mask.clone()

    return key_indexs

    
def add_stride_key_index(T, key_indexs, key_stride=3):


    skey_indexs = [] 
    for i in range(0,T, key_stride):
        if i in key_indexs:
            continue 

        save = True
        for j in range(i-key_stride//2, i+key_stride//2):
            if j in key_indexs:
                save = False

        if save:
            skey_indexs.append(i)

    
    if T-1 not in skey_indexs and T-1 not in key_indexs:
        skey_indexs.append(T-1)


    ref_indexs, _ = find_ref_key_indexs(key_indexs, key_indexs = skey_indexs)
    ref_indexs = ref_indexs[0]

    for i in ref_indexs:
        key_indexs.append(i)
    
    return key_indexs



def resize_flow(flow, shape):

    _,_,h,w = flow.size()

    h_ratio, w_ratio = shape[0]/h, shape[1]/w 
    resized_flow =  F.interpolate(flow, size=shape)

    resized_flow[:,0] = resized_flow[:,0] * w_ratio
    resized_flow[:,1] = resized_flow[:,1] * h_ratio

    return resized_flow
        
        
def dilation(image, ksize=3, iter=1):
    result = image
    for i in range(iter):
        image_pad = F.pad(result.detach(), [ksize//2, ksize//2, ksize//2, ksize//2], mode='constant', value=0)
        # Unfold the image to be able to perform operation on neighborhoods
        image_unfold = F.unfold(image_pad, kernel_size=(ksize,ksize))
        sums = image_unfold 
        # Take maximum over the neighborhood
        result, _ = sums.max(dim=1)
        # Reshape the image to recover initial shape
        result = torch.reshape(result, image.shape)
    return result
