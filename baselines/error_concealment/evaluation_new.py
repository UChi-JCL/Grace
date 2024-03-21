import os
import cv2
import glob
import importlib
import numpy as np
import scipy.ndimage
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

from model.RAFT import RAFT

import error_concealment.utils as utils
from libs.video_quality import ssim
from libs import PerceptualSimilarity


class Trainer():
    def __init__(self,args):
        MODEL_DIR = "models/error_concealment"

        self.args = args
        self.device = self.args.device

        # Flow Estimator
        self.netF = torch.nn.DataParallel(RAFT(self.args)).to(self.device)
        model_weight = torch.load(f'{MODEL_DIR}/FlowEstimator.pth')
        self.netF = self.netF.module
        self.netF.load_state_dict(model_weight['netF'], strict=True)
        self.netF = torch.nn.DataParallel(self.netF).to(self.device)

        
        # Error Compensation network
        net = importlib.import_module('model.'+self.args.model)
        self.netE = net.ErrorCompensationNetwork(self.args)
        pretrained_model = '{}'.format(f'{MODEL_DIR}/ECN.pth')
        model_weight = torch.load(pretrained_model)
        self.netE = torch.nn.DataParallel(self.netE).to(self.device)
        self.netE.load_state_dict(model_weight['netE'], strict=True) 

        # Local Temporal Network
        # We use local temporal network as light version of FuseFormer (stack_num 4)
        ff = importlib.import_module('model.fuseformer')
        self.netC = ff.InpaintGenerator(stack_num=4)
        data = torch.load(f'{MODEL_DIR}/LTN.pth')
        self.netC.load_state_dict(data['netC'])
        self.netC = torch.nn.DataParallel(self.netC).to(self.device)

        # Synthesis network. We use FuseFormer.
        self.netS = ff.InpaintGenerator(stack_num=8)
        data = torch.load(f'{MODEL_DIR}/fuseformer.pth')
        self.netS.load_state_dict(data)
        self.netS = torch.nn.DataParallel(self.netS).to(self.device)
        

        # For measuring perceptual similarity
        self.dm_model = PerceptualSimilarity.dist_model.DistModel()
        self.dm_model.initialize(model='net-lin', net='alex', use_gpu=True)

        self.netE.eval()
        self.netS.eval()
        self.netC.eval()
        self.netF.eval()


    
    def prepare_data(self, frame_path, mask_path, test_data_name, stationary_mask_path=None):
    
        # Read imgs and frames 
        v_frames = sorted(glob.glob(frame_path + test_data_name + '/*'))
        v_masks = sorted(glob.glob(mask_path + test_data_name + '/*'))            
        
        # Obtains original imgH, imgW
        oimgH, oimgW =  np.array(Image.open(v_frames[0])).shape[:2]

        # Crop frames for forwarding RAFT
        sf = 32
        cropH, cropW = oimgH%sf, oimgW%sf
        # imgH, imgW = oimgH - cropH, oimgW-cropW
        cropU, cropD = cropH//2, cropH -cropH//2
        cropL, cropR = cropW//2, cropW -cropW//2

        # Load video.
        video = []
        for filename in v_frames:
            frame = Image.open(filename)
            frame = np.array(frame).astype(np.uint8)
            frame = frame[cropU:oimgH-cropD, cropL:oimgW-cropR]
            video.append(torch.from_numpy(frame).permute(2, 0, 1).float())
            

        # Load mask
        mask = []
        flow_mask = []
        for filename in v_masks:

            if self.args.mask_mode == 1:
                mask_img = Image.open(stationary_mask_path).convert('L')
            else:
                mask_img = Image.open(filename).convert('L')

            mask_img = np.array(mask_img)

            if self.args.davis:

                ################## Same setting with FGVC ##########################
                # Dilate 15 pixel so that all known pixel is trustworthy
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
                # Close the small holes inside the foreground objects
                flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool)
                flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.bool)
                flow_mask_img = flow_mask_img[cropU:oimgH-cropD, cropL:oimgW-cropR]
                flow_mask_img = torch.from_numpy(flow_mask_img).float().unsqueeze(0)

                # Dilate 5 pixel so that all known pixel is trustworthy
                # Close the small holes inside the foreground objects
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=9)
                mask_img = scipy.ndimage.binary_fill_holes(mask_img).astype(np.bool)
                mask_img = mask_img[cropU:oimgH-cropD, cropL:oimgW-cropR]
                mask_img = torch.from_numpy(mask_img).float().unsqueeze(0)

            else:
                mask_img = mask_img[cropU:oimgH-cropD, cropL:oimgW-cropR]
                mask_img = torch.from_numpy(mask_img).float().unsqueeze(0)
                flow_mask_img = mask_img.clone()

            mask.append(mask_img)
            flow_mask.append(flow_mask_img)
    

        frames = torch.stack(video, dim=0).to(self.device) # T, 3, H, W
        masks = torch.stack(mask, dim=0).to(self.device) # T, 1, H, W
        flow_masks = torch.stack(flow_mask, dim=0).to(self.device) # T, 1, H, W

        frames = frames/255.
        frames = frames*2 -1  # normalize -1 to 1

        # Resize input frames and masks to the same setting with STTN and FuseFormer
        frames = F.interpolate(frames, size=(self.args.test_h,self.args.test_w))
        masks = F.interpolate(masks, size=(self.args.test_h,self.args.test_w))
        flow_masks = F.interpolate(flow_masks, size=(self.args.test_h,self.args.test_w))

        return frames, masks, flow_masks



    def propagate(self, tar_frame, tar_mask, ref_frame, ref_mask, flow):
        # Using the completed flows, warp reference_frame to target_frame and propagate pixels
        

        # Find the valid regions at warped_reference_frame
        # Due to flow warping, especially for border part, first find vliad regions using flow    
        flow_valid_mask = torch.ones_like(ref_mask).to(self.device)
        flow_valid_mask = utils.warp(flow_valid_mask, flow)
        flow_valid_mask = (flow_valid_mask >0.5).float()

        # Warping reference mask to target index
        warped_ref_mask = ((utils.warp(ref_mask, flow)>0.5).float()) * flow_valid_mask
        warped_ref_mask = utils.dilation(warped_ref_mask, iter=1)
        ref_valid_mask = (1-warped_ref_mask) * flow_valid_mask

        # Find matched(corresponding) regions in reference frame
        warped_ref_frame = utils.warp(ref_frame, flow)
        warped_ref_frame = warped_ref_frame*(1-warped_ref_mask)

        # Get propagation and remaining mask
        prop_mask = ((ref_valid_mask * tar_mask)>0.5).float()  # propagation mask
        rem_mask = ((tar_mask-prop_mask)>0.5).float() # remainig mask
                
        # Enlarge original mask (Dilate hole region)
        tar_dmask = ((utils.dilation(tar_mask, iter=self.args.dilation_iter))>0.5).float()
        # Remove the regions where the pixels cannot propagate (outside the flow_valid_mask)
        tar_dmask = ((tar_dmask * flow_valid_mask)>0.5).float()

        # Where to propagate more pixels than filled frame
        dprop_mask = ((ref_valid_mask * tar_dmask)>0.5).float() 

        # Get filled_frame and overfilled_frame
        filled_frame = tar_frame*(1-prop_mask) + warped_ref_frame*prop_mask
        ofilled_frame = tar_frame*(1-dprop_mask) + warped_ref_frame*dprop_mask

    
        return ofilled_frame, filled_frame, warped_ref_frame, warped_ref_mask, prop_mask, rem_mask, flow_valid_mask, tar_dmask

       
    

    def generate_error_guidance(self, filled_frame, ofilled_frame, mask, dmask, prop_mask, rem_mask, warped_ref_mask, flow_valid_mask):

        # Groundtruth error for training
        # For training, err_gt = GT(filled_frame) - ofilled_frame
        err_gt = filled_frame - ofilled_frame
        
        # Estimate error mask
        # Find only the valid regions from reference frames
        tmp_mask = ((warped_ref_mask + mask)>0.5).float()
        tmp_mask = ((flow_valid_mask - tmp_mask)>0.5).float()

        # Get error mask. dmask(Dilated_mask) - mask(original_target_mask)
        err_mask = (((dmask-mask)>0.5).float()) * tmp_mask
        # Get error guidance map
        err_guidance = err_gt * err_mask

        # This is only for training        
        err_gt = err_gt * flow_valid_mask * (dmask)

        '''
        If the target hole can be fully filled by propagation at once, error mask comes out in the shape of a band surrounding the mask. 
        (Like the Figure5 in our main paper)
        But, in some reasons (due to flow warping or out of border), error mask partial surrounds the original mask.
        In order to inform the network about this region, we define it as 'empty_error_mask'.
        
        This mask is not introduced in our paper for simplicty.
        
        '''
        
        merge_mask = (((err_mask + rem_mask + prop_mask) > 0.5).float()) * flow_valid_mask
        empty_err_mask = ((dmask - merge_mask)>0.5).float()
       
        return err_mask, err_guidance, empty_err_mask, err_gt


    def compensate(self, filled_frames, ofilled_frames,  err_pixels, rem_masks, err_masks, prop_masks, empty_err_masks):

        with torch.no_grad():    
            if torch.sum(err_masks) > 0: 
                err_outs = self.netE(ofilled_frames, err_pixels, rem_masks, err_masks, prop_masks, empty_err_masks) 
            else:
                err_outs = torch.zeros_like(filled_frames).cuda()
            
            compensated_frames = prop_masks*(err_outs+filled_frames) + (1-prop_masks)*filled_frames

        return err_outs, compensated_frames



    def synthesis(self, refined_frame, rem_mask):

        if torch.sum(rem_mask) > 1:
            with torch.no_grad():
                pred = self.netS(refined_frame.unsqueeze(0))
        else: # if there is no hole part, skip the synthesis network
            pred = refined_frame 

        pred = refined_frame * (1-rem_mask) + pred * rem_mask

        return pred


    def inpainting(self, tar_indexs, masks, masked_frames, ref_indexs, flowFs, flowBs, min_ref_len):


        max_len = self.args.max_len

        # Nearest frames first
        for i in range(min_ref_len):

            # If the tar_indexs is too long, split into small temporal window
            for idx_start in range(0, len(tar_indexs), max_len): 
                if idx_start + max_len > len(tar_indexs):
                    idx_end = len(tar_indexs)
                else:
                    idx_end = idx_start + max_len

                ofilled_frames, filled_frames, ref_frames, err_guidances, = [], [], [], []
                rem_masks, prop_masks, err_masks, empty_err_masks  = [], [], [], []

                
                for l, tar_idx in enumerate(tar_indexs[idx_start:idx_end]): 
                    
                    tar_frame = masked_frames[tar_idx:tar_idx+1].clone()
                    tar_mask = masks[tar_idx:tar_idx+1].clone()
                    
                    # Find one ref_index for tar_index, and get corresponding masked_frame and mask
                    ref_idx = ref_indexs[l][i]
                    ref_frame = masked_frames[ref_idx:ref_idx+1]
                    ref_mask = masks[ref_idx:ref_idx+1]

                    # Trace flow form tar_idx to ref_idx 
                    flow = utils.trace_all_flows(flowFs, flowBs, tar_idx, ref_idx)
                    

                    ############################################
                    ############### Propagation ################
                    ############################################
                    # Propagte pixels from  reference_frame to target_frame
                    ofilled_frame, filled_frame, warped_ref_frame, warped_ref_mask, prop_mask, rem_mask, flow_valid_mask, tar_dmask \
                         = self.propagate(tar_frame, tar_mask, ref_frame, ref_mask, flow)
        
                    
                    # Get err gudiance map and other masks
                    err_mask, err_guidance, empty_err_mask,  _ = \
                        self.generate_error_guidance(filled_frame, ofilled_frame, tar_mask, tar_dmask, prop_mask, rem_mask, warped_ref_mask, flow_valid_mask)
                            

                    ofilled_frames.append(ofilled_frame)
                    filled_frames.append(filled_frame)
                    ref_frames.append(warped_ref_frame)
                    err_guidances.append(err_guidance)

                    rem_masks.append(rem_mask)
                    prop_masks.append(prop_mask)
                    empty_err_masks.append(empty_err_mask)
                    err_masks.append(err_mask)
                    
                    
                ofilled_frames = torch.stack(ofilled_frames, 1)    
                filled_frames = torch.stack(filled_frames, 1)
                ref_frames = torch.stack(ref_frames, 1)
                err_guidances = torch.stack(err_guidances, 1)


                rem_masks = torch.stack(rem_masks, 1)
                prop_masks = torch.stack(prop_masks, 1)
                empty_err_masks = torch.stack(empty_err_masks, 1)    
                err_masks = torch.stack(err_masks, 1)
                

                ############################################
                ############## Compensation ################
                ############################################

                if torch.sum(prop_masks) == 0 :
                    # If there are noting to propagate ... 
                    err_outs = err_guidances.clone()
                    compensated_frames = filled_frames.clone()
                    # continue
                else:
                    err_outs, compensated_frames = self.compensate(filled_frames, ofilled_frames, \
                          err_guidances, rem_masks, err_masks, prop_masks, empty_err_masks)


                # Update compensated results
                for t, tar_idx in enumerate(tar_indexs[idx_start:idx_end]):
                    masked_frames[tar_idx:tar_idx+1] = compensated_frames[:,t]
                    masks[tar_idx:tar_idx+1] = rem_masks[:,t]

            
                # If there is no hole anymore, stop iteration
                if torch.sum(rem_masks) == 0:
                    break

    def flow_completion(self, frames, masks, flow_masks):
        T,_,_,_ = frames.size()
    
        masks = (masks>0.5).float()
        flow_masks = (flow_masks>0.5).float()
        masked_frames = frames*(1-masks)

        neighbor_len = self.args.neighbor_len
        coarse_preds= [None] * T
        
        ############################################
        ########## Local Temporal Network ##########
        ############################################
        for f in range(0, T, neighbor_len):
            neighbor_ids = [i for i in range(max(0, f-neighbor_len), min(T, f+neighbor_len+1))]
            with torch.no_grad():
                coarse_pred = self.netC(masked_frames[neighbor_ids].unsqueeze(0))
                coarse_pred = masked_frames[neighbor_ids]*(1-masks[neighbor_ids]) + coarse_pred*(masks[neighbor_ids]) # TCHW        
                
            for i, idx in enumerate(neighbor_ids):
                if coarse_preds[idx] is None:
                    coarse_preds[idx] = coarse_pred[i]
                else:
                    coarse_preds[idx] = coarse_pred[i]*0.5 + coarse_preds[idx]*0.5 # blending 
    
        coarse_preds = torch.stack(coarse_preds, 0)


        # For flow estimator (flow completion), upscaling the input frames works better.
        # The size will be reduced after getting completed flows
        # In our experiments, we upsample by 2
    
        frames = F.interpolate(frames, size=(self.args.test_h*2,self.args.test_w*2))
        coarse_preds = F.interpolate(coarse_preds, size=(self.args.test_h*2,self.args.test_w*2))
        masks = F.interpolate(masks, size=(self.args.test_h*2,self.args.test_w*2))
        flow_masks = F.interpolate(flow_masks, size=(self.args.test_h*2,self.args.test_w*2))
        masks = (masks>0.5).float()
        flow_masks = (flow_masks>0.5).float()

        ############################################
        ############## Flow Estimator ##############
        ############################################

        videoFlowF = []
        videoFlowB = [] 

        # Forward flows
        for i in range(0, T-1):
            with torch.no_grad():
                #if self.args.davis:
                #    _, completed_flow = self.netF(coarse_preds[i:i+1], coarse_preds[i+1:i+2], iters=20,  mask1=flow_masks[i:i+1], mask2=flow_masks[i+1:i+2], test_mode=True)                
                #else:
                #   _, completed_flow = self.netF(coarse_preds[i:i+1], coarse_preds[i+1:i+2], iters=20, mask1=masks[i:i+1], mask2=masks[i+1:i+2], test_mode=True)
                _, completed_flow = self.netF(coarse_preds[i:i+1], coarse_preds[i+1:i+2], iters=20, mask1=masks[i:i+1], mask2=masks[i+1:i+2], test_mode=True)
            videoFlowF.append(completed_flow)

        # Backward flows
        for i in range(T-1, 0, -1):
            with torch.no_grad():
                #if self.args.davis:
                #    _, completed_flow = self.netF(coarse_preds[i:i+1], coarse_preds[i-1:i], iters=20,  mask1=flow_masks[i:i+1], mask2=flow_masks[i-1:i], test_mode=True)
                #else:
                #    _, completed_flow = self.netF(coarse_preds[i:i+1], coarse_preds[i-1:i], iters=20, mask1=masks[i:i+1], mask2=masks[i-1:i], test_mode=True)
                _, completed_flow = self.netF(coarse_preds[i:i+1], coarse_preds[i-1:i], iters=20, mask1=masks[i:i+1], mask2=masks[i-1:i], test_mode=True)
            videoFlowB.append(completed_flow)

        # Revese order for backward flows
        videoFlowB = videoFlowB[::-1]
        
        # Concat all the completed flows    
        videoFlowF = torch.cat(videoFlowF, 0 ) # T-1 2 test_h test_W
        videoFlowB = torch.cat(videoFlowB, 0 ) # T-1 2 test_h test_W
        
        # Go back to original size
        with torch.no_grad():
            videoFlowF = utils.resize_flow(videoFlowF, shape=(self.args.test_h, self.args.test_w))
            videoFlowB = utils.resize_flow(videoFlowB, shape=(self.args.test_h, self.args.test_w))

        return videoFlowF, videoFlowB

    

    

    def eval(self, ref_frame, new_frame, mask):
        """
        Input:
            ref_frame, new_frame: 3, h, w tensor
            mask: the frame mask, 1 means remove, 0 means keep
        Output:
            inpainted new frame
        """
    
        ## Read original frames and corresponding masks 
        #

        #if self.args.davis:
        #    # data_path = './datasets/davis/'
        #    # frame_path = data_path + 'JPEGImages/'
        #    # mask_path = data_path + 'Annotations/'
        #    # data_name = 'davis'

        #    data_name = self.args.data_name 
        #    # data_path = '../FGVC/data/' + data_name + '/'
        #    data_path = self.args.test_data_root + data_name + '/'   
        #    frame_path = data_path + 'JPEGImages/480p/'
        #    mask_path = data_path + 'Annotations/480p/'
        #else:
        #    data_name = self.args.data_name 
        #    # data_path = '../STTN/datasets/youtube-vos/' + data_name + '/'         
        #    data_path = self.args.test_data_root + data_name + '/'   
        #    frame_path = data_path + 'frames/'
        #    mask_path = data_path + 'masks/'

        #    stationary_mask_path = data_path + 'stationary_mask.png'

        #video_name_list = sorted(glob.glob(os.path.join(mask_path, '*')))

        #

        #####################################################################################
        #####################################################################################
        #####################################################################################
        #####################################################################################
        #####################################################################################

        total_psnrs, total_ssims, total_lpips = [], [], []

        v_psnrs, v_ssims, v_lpips = [], [], []



        frames = torch.stack([ref_frame, new_frame])
        T,_,_,_ = frames.size()
        ref_mask = torch.zeros_like(mask).to(mask.device)
        masks = torch.stack([ref_mask, mask])
        masks = masks[:, 0:1, :, :]
        flow_masks = masks.clone()

        ############################################
        ######### Step 1 : Flow completion #########
        ############################################
        videoFlowF, videoFlowB = self.flow_completion(frames, masks, flow_masks)

        
        frames = F.interpolate(frames, size=(self.args.test_h,self.args.test_w))
        masks = F.interpolate(masks, size=(self.args.test_h,self.args.test_w))

        masks = (masks>0.5).float()
        masked_frames = frames *(1-masks)

        

    
        ############################################
        ########### Step 2 : Inpainting ############
        ############################################
        
        

        ############################################
        ######## Step 2.1 : Find key frames ########
        ############################################
        
        key_idxs = utils.find_key_indexs(masks, videoFlowF, videoFlowB, stride=1)
        
        # Find reference frames for each of key_indexs
        # ex, Key_idx : 10 --> ref_idx : [9,11,8,12,7,13, ...]
        # We set different stride factor for object removal and video restoration scenarios
        if self.args.davis:
            ref_idxs, min_ref_len = utils.find_ref_indexs(key_idxs, T=T, stride=5)
        else:
            ref_idxs, min_ref_len = utils.find_ref_indexs(key_idxs, T=T, stride=3)

        
        

        ############################################
        ###### Step 2.2 : Inpaint key frames ######
        ############################################
        
        zero_mask = torch.zeros_like(masks[0:1])

        # Propagate from reference frames and compensate errors for key frames
        self.inpainting(key_idxs, masks, masked_frames, ref_idxs, videoFlowF, videoFlowB, min_ref_len)

            
        # If there are remaining holes in key frames, do synthesis
        if torch.sum(masks[key_idxs]) > 0 :
                
            neighbor_len = self.args.neighbor_len
            for kidx in key_idxs:
                neighbor_idxs = [i for i in range(max(0, kidx-neighbor_len), min(T, kidx+neighbor_len+1))]
                for n, nidx in enumerate(neighbor_idxs):
                    if nidx == kidx:
                        output_idx = n
                    
                outputs = self.synthesis(masked_frames[neighbor_idxs], masks[neighbor_idxs]) 

                # Save the completed key frames and corresponding masks
                masked_frames[kidx] = outputs[output_idx].clone()
                masks[kidx] = zero_mask.clone() # Fill masks (Set 0)

                

        ############################################
        #### Step 2.3 : Inpaint non-key frames ####
        ############################################
        
        # Find index of non-key frames
        non_key_idxs = []
        for t in range(0,T):
            if t in key_idxs:
                continue
            non_key_idxs.append(t)

        
        nk_stride = self.args.non_key_len
        for l in range(0, len(non_key_idxs), nk_stride):
            
            if l+nk_stride >= len(non_key_idxs):
                nk_idxs = non_key_idxs[l:]
            else:
                nk_idxs = non_key_idxs[l:l+nk_stride]

            # Find nearest key indexes from given non-key index 
            ref_idxs, min_ref_len = utils.find_ref_key_indexs(nk_idxs, key_indexs = key_idxs)
            
            
            # Set two directions for non-key frame
            # only the first two indexes in key idxs are exchanged 
            # ex) target non-key index : 25, key_frames : [0,10,30,50]
            # ref_idxs : [30,10,50,0], rev_ref_idxs : [30,10,50,0]
            rev_ref_idxs = []
            for ridx in ref_idxs:
                tmp_ridx = ridx[:]
                tmp_ridx[0], tmp_ridx[1] = tmp_ridx[1], tmp_ridx[0]
                rev_ref_idxs.append(tmp_ridx)

        
            rev_masks = masks.clone()
            rev_masked_frames = masked_frames.clone()

            
            # Inapinting forward and backward direction from given key_indexes
            self.inpainting(nk_idxs, masks, masked_frames, ref_idxs, videoFlowF, videoFlowB, min_ref_len)
            self.inpainting(nk_idxs, rev_masks, rev_masked_frames, rev_ref_idxs, videoFlowF, videoFlowB, min_ref_len)
        
            
            for n, nidx in enumerate(nk_idxs):
                # Blend according to temporal length weights
                blend_1 = abs(nidx - ref_idxs[n][0])
                blend_2 = abs(nidx - rev_ref_idxs[n][0])
                blend_len = blend_1 + blend_2
                blend_1 = (blend_len - blend_1)/blend_len
                blend_2 = (blend_len - blend_2)/blend_len

                output1 = masked_frames[nidx:nidx+1].clone()
                output2 = rev_masked_frames[nidx:nidx+1].clone()
                output = blend_1*output1 + blend_2*output2
                
                # Save results
                masked_frames[nidx:nidx+1] = output.clone()
                masks[nidx:nidx+1] = zero_mask.clone()

                        
        return masked_frames[1]

        # Calculate PSNR,SSIM values and save results
        #for idx in range(T):
        #    GT = frames[idx:idx+1].clone()                
        #    output = masked_frames[idx:idx+1].clone()
        #    
        #    np_GT = utils.tensor_to_numpy(GT, denorm=True)[0] # [0~255]
        #    np_output = utils.tensor_to_numpy(output, denorm=True)[0] # [0~255]
        #
        #    v_psnrs.append(utils.psnr_measure(np_output, np_GT))
        #    v_ssims.append(ssim.ssim_exact(np_output / 255, (np_GT) / 255))

        #    gt_tensor = PerceptualSimilarity.util.util.im2tensor((np_GT)[..., :3])
        #    result_tensor = PerceptualSimilarity.util.util.im2tensor(np_output[..., :3])
        #    p_dist = self.dm_model.forward(gt_tensor, result_tensor)
        #    p_dist = p_dist.item()
        #    v_lpips.append(p_dist)
        #    
        #del videoFlowF
        #del videoFlowB
        #    
        #return output
        #                

        ## Empty memory

        #torch.cuda.empty_cache()

        ## Average evaluation metrics
        #v_psnrs = np.mean(np.array(v_psnrs))
        #v_ssims = np.mean(np.array(v_ssims))
        #v_lpips = np.mean(np.array(v_lpips))

        #total_psnrs.append(v_psnrs)
        #total_ssims.append(v_ssims)
        #total_lpips.append(v_lpips)

        #if self.args.davis:
        #    print('{}'.format(test_data_name))
        #else:
        #    print('{}-{} : psnr-{:.4f} ssim-{:.4f}, lpips-{:.4f}'.format("test", "Test", v_psnrs, v_ssims, v_lpips))


        # Print out evaluation values
        #if self.args.davis is False:
        #    print('====================================================')
        #    excel_line = []

        #    for v in range(len(video_name_list)):
        #        test_data_name = video_name_list[v].split('/')[-1]                    
        #        excel_line.append([test_data_name, total_psnrs[v], total_ssims[v], total_lpips[v]])
        #        
        #    total_psnrs = np.mean(np.array(total_psnrs))
        #    total_ssims = np.mean(np.array(total_ssims))
        #    total_lpips = np.mean(np.array(total_lpips))
        #    
        #    
        #    print('final total psnrs for {} : {:.4f} : {:.4f} : {:.4f}'.format(data_name, total_psnrs, total_ssims, total_lpips))
        #    excel_line.append(['total', total_psnrs, total_ssims, total_lpips])

        #    df = pd.DataFrame(excel_line, columns=['video', 'psnrs', 'ssim', 'lpips'])
        #    
        #    if self.args.mask_mode == 0:
        #        df.to_excel('ECFVI_' + data_name + '_moving.xlsx', index=False) 
        #    else:
        #        df.to_excel('ECFVI_' + data_name + '_stationary.xlsx', index=False) 
    


     
