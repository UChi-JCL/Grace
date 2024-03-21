## Academic Software License: © 2023 UChicago (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-4:
## 
## Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.
## Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.
## IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

from audioop import avg, avgpp
from statistics import mode
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import ctypes
import copy
import json
import pandas as pd
import pandas as pd
#from cabac_coder.cabac_coder import CABACCoder, CABACCoderTorchWrapper
import os, sys
import subprocess as sp
import io
import shlex
import cv2
from copy import deepcopy
from tqdm import tqdm
from autoencoder_interface import *
from streaming_interface import *
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
from skimage.metrics import peak_signal_noise_ratio
import time
# from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from queue import PriorityQueue
from dataclasses import dataclass
import PIL
import random
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


df_psnr = None

def PSNR(Y1_raw, Y1_com):
    Y1_com = Y1_com.to(Y1_raw.device)
    log10 = torch.log(torch.FloatTensor([10])).squeeze(0).to(Y1_raw.device)
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    quality = 10.0*torch.log(1/train_mse)/log10
    return float(quality)

def SSIM(Y1_raw, Y1_com):
    #y1 = Y1_raw.permute([1,2,0]).cpu().detach().numpy()
    #y2 = Y1_com.permute([1,2,0]).cpu().detach().numpy()
    #return ssim(y1, y2, multichannel=True)
    return float(ssim( Y1_raw.float().cuda().unsqueeze(0), Y1_com.float().unsqueeze(0), data_range=1, size_average=False).cpu().detach()) 

def PSNR_YUV(yuv1, yuv2):
    mse = np.mean((yuv1 - yuv2) ** 2)
    max_pixel = max(np.max(yuv1), np.max(yuv2))
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr

def SSIM_YUV(y1, y2):
    return ssim(y1, y2, multichannel=False)

def rgb_tensor_to_img(rgbtensor):
    return np.array(to_pil_image(rgbtensor.clip(0, 1)))

def RGB2YUV(rgb, isTensor):
    """
    rgb: numpy array in (h, w, c)
    """
    if isTensor:
        rgb = rgb_tensor_to_img(rgb)
    yvu = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, v, u = cv2.split(yvu)
    u = cv2.resize(u, (u.shape[1]//2, u.shape[0]//2))
    v = cv2.resize(v, (v.shape[1]//2, v.shape[0]//2))
    return y, u, v, np.concatenate((y,u,v), axis=None)

def metric_all_in_one(Y1_raw, Y1_com):
    """
    returns: 
        rgbpsnr, rgbssim, yuvpsnr, yuvssim
    """
    rgbpsnr = PSNR(Y1_raw, Y1_com)
    # breakpoint()
    rgbssim = float(ssim( Y1_raw.float().cuda().unsqueeze(0), Y1_com.float().unsqueeze(0), data_range=1, size_average=False).cpu().detach()) 

    # y1, u1, v1, yuv1 = RGB2YUV(Y1_raw, True)
    # y2, u2, v2, yuv2 = RGB2YUV(Y1_com, True)

    # yuvpsnr = PSNR_YUV(yuv1, yuv2)
    # yuvssim = SSIM_YUV(y1, y2)
    return float(rgbpsnr), rgbssim, 0, 0

def FFMPEG_PSNR(enc_frames, raw_frames, outfile):
    """
    frames: frames in torch tensor C,H,W format
    raw_video: the name of raw_video
    """
    def get_output_folder():
        output_filename = f'/tmp/output-{np.random.randint(0, 100000)}-folder'
        while os.path.exists(output_filename):
            output_filename = f'/tmp/output-{np.random.randint(0, 100000)}-folder'
        os.makedirs(output_filename, exist_ok=True)
        return output_filename

    def free_tmp_folder(outfile):
        os.system("rm -rf {}".format(outfile))

    outfolder = get_output_folder()
    print("The folder is", outfolder)
    for idx, frame in tqdm(enumerate(enc_frames)): 
        save_image(frame, os.path.join(outfolder, f"enc-{idx:03d}.png"))
    for idx, frame in tqdm(enumerate(raw_frames)): 
        save_image(frame, os.path.join(outfolder, f"raw-{idx:03d}.png"))
    
    cmd = f"ffmpeg -i {outfolder}/enc-%03d.png -crf 0 {outfolder}/enc.mp4"
    os.system(cmd)
    cmd = f"ffmpeg -i {outfolder}/raw-%03d.png -crf 0 {outfolder}/raw.mp4"
    os.system(cmd)

    os.system(f"ffmpeg -i {outfolder}/enc.mp4 -i {outfolder}/raw.mp4 -lavfi psnr=stats_file={outfile}.psnr -f null -")
    os.system(f"ffmpeg -i {outfolder}/enc.mp4 -i {outfolder}/raw.mp4 -lavfi ssim=stats_file={outfile}.ssim -f null -")

    free_tmp_folder(outfolder)


def get_block_psnr(frame_id, gt_frame, dec_frame, w_step, h_step):
    """
    return frame_id, blk_id, psnr
    """
    C, H, W = dec_frame.shape
    psnrs = []
    for h in range(0, H, h_step):
        for w in range(0, W, w_step):
            gt_clip = gt_frame[:, h:h+h_step, w:w+w_step]
            dec_clip = dec_frame[:, h:h+h_step, w:w+w_step]
            value = PSNR(gt_clip, dec_clip)
            psnrs.append(float(value))
    ret = pd.DataFrame()
    ret["psnr"] = psnrs
    ret["frame_id"] = frame_id
    ret["block_id"] = ret.index
    return ret


METRIC_FUNC = PSNR

def read_video_into_frames(video_path, frame_size = None, nframes=1000):
    """
    Input:
        video_path: the path to the video
        frame_size: resize the frame to a (width, height), if None, it will not do resize
        nframes: number of frames
    Output:
        frames: a list of PIL images
    """
    def create_temp_path():
        path = f"/tmp/yihua_frames-{np.random.randint(0, 1000)}/"
        while os.path.isdir(path):
            path = f"/tmp/yihua_frames-{np.random.randint(0, 1000)}/"
        os.makedirs(path, exist_ok=True)
        return path

    def remove_temp_path(tmp_path):
        os.system("rm -rf {}".format(tmp_path))

    frame_path = create_temp_path()
    if frame_size is None:
        cmd = f"ffmpeg -i {video_path} {frame_path}/%03d.png 2>/dev/null 1>/dev/null"
        #cmd = f"ffmpeg -i {video_path} {frame_path}/%03d.png"
    else:
        width, height = frame_size
        cmd = f"ffmpeg -i {video_path} -s {width}x{height} {frame_path}/%03d.png 2>/dev/null 1>/dev/null"

    print(cmd)
    os.system(cmd)
    
    image_names = os.listdir(frame_path)
    frames = []
    for img_name in sorted(image_names)[:nframes]:
        frame = Image.open(os.path.join(frame_path, img_name))

        ''' pad to nearest 64 for Grace model '''
        padsz = 128
        w, h = frame.size
        pad_w = int(np.ceil(w / padsz) * padsz)
        pad_h = int(np.ceil(h / padsz) * padsz)

        frames.append(frame.resize((pad_w, pad_h)))

    print(f"frame path is: {frame_path}")
    print(f"Got {len(image_names)} image names and {len(frames)} frames")
    print("frameSize", len(frames))
    print("Resizing image to", frames[0].size)
    remove_temp_path(frame_path)
    return frames

def read_video_into_frames_opencv(video_path, frame_size=None, nframes=1000):
    """
    Input:
        video_path: the path to the video
        frame_size: resize the frame to a (width, height), if None, it will not do resize
        nframes: number of frames
    Output:
        frames: a list of PIL images
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        if np.sum(img) == 0:
            continue

        img = Image.fromarray(img)
        if frame_size is not None:
            img = img.resize(frame_size)
        else:
            ''' pad to nearest 64 '''
            padsz = 64
            w, h = img.size
            pad_w = int(np.ceil(w / padsz) * padsz)
            pad_h = int(np.ceil(h / padsz) * padsz)
            img = img.resize((pad_w, pad_h))
        frames.append(img)

        if len(frames) >= nframes:
            break
    print("Resizing image to", frames[-1].size)
    return frames


lib = ctypes.CDLL("libs/bpgenc.so")
lib2 = ctypes.CDLL("libs/bpgdec.so")
bpg_encode_bytes = lib.bpg_encode_bytes
bpg_decode_bytes = lib2.bpg_decode_bytes
get_buf = lib.get_buf
get_buflen = lib.get_buf_length
free_mem = lib.free_memory
get_buf.restype = ctypes.POINTER(ctypes.c_char)
bpg_decode_bytes.restype = ctypes.POINTER(ctypes.c_char)

def bpg_encode(img):
    frame = (torch.clamp(img, min = 0, max = 1) * 255).round().byte()
    _, h, w = frame.shape
    frame2 = frame.permute((1, 2, 0)).flatten()
    bs = frame2.numpy().tobytes()
    ubs = (ctypes.c_ubyte * len(bs)).from_buffer(bytearray(bs))
    bpg_encode_bytes(ubs, h, w)
    buflen =  get_buflen()
    buf = get_buf()
    bpg_stream = ctypes.string_at(buf, buflen)
    free_mem(buf)
    return bpg_stream, h, w, len(bpg_stream)

def bpg_decode(bpg_stream, h, w):
    ub_result = (ctypes.c_ubyte * len(bpg_stream)).from_buffer(bytearray(bpg_stream))
    rgb_decoded = bpg_decode_bytes(ub_result, len(bpg_stream), h, w)
    b = ctypes.string_at(rgb_decoded, h * w * 3)
    bytes = np.frombuffer(b, dtype=np.byte).reshape((h, w, 3))
    image = torch.tensor(bytes).permute((2, 0, 1)).byte().float().cuda()
    image = image / 255
    free_mem(rgb_decoded)
    return image

class IPartFrame:
    def __init__(self, code, shapex, shapey, offset_width, offset_height):
        self.code = code
        self.shapex = shapex
        self.shapey = shapey
        self.offset_width = offset_width
        self.offset_height = offset_height

class EncodedFrame:
    """
    self.code is torch.tensor
    """
    def __init__(self, code, shapex, shapey, frame_type, frame_id):
        self.code = code
        self.shapex = shapex
        self.shapey = shapey
        self.frame_type = frame_type
        self.frame_id = frame_id
        self.loss_applied = False
        self.ipart = None
        self.isize = None
        self.tot_size = None

    def apply_loss(self, loss_ratio, blocksize = 100):
        """
        default block size is 100
        """
        if isinstance(self.code, AECode):
            self.code.get_tensor("res").simulate_loss(loss_ratio, blocksize)
            self.code.get_tensor("mv").simulate_loss(loss_ratio, blocksize)
        
        if self.ipart is not None and np.random.random() < loss_ratio:
            self.ipart = None

def find_mn_from_ab(a, b):
    """
    return m, n such that a = mp, b = nq and p > 1, q > 1 and mn = {10, 12, 8, 15, 6}
    """
    mnlist = [(2, 5), (5, 2), (10, 1), (1, 10), 
              (2, 6), (6, 2), (3, 4), (4, 3), (1, 12), (12, 1), 
              (3, 5), (5, 3), (2, 3), (3, 2), (1, 6), (6, 1)]
    for m, n in mnlist:
        if a % m == 0 and a // m > 1 and b % n == 0 and b // n > 1:
            return m, n
    raise RuntimeError(f"No suitable m, n found for a, b = {a}, {b}")

def set_hw_step(h, w):
    """
    returns h_step and w_step
    """
    a, b = h // 64, w // 64
    m, n = find_mn_from_ab(a, b)
    return h // m, w // n

class AEModel:
    def __init__(self, ae_coder: AEAdapter):
        self.ae_coder = ae_coder
        self.entropy_coder = EntropyCodec()

        self.reference_frame = None
        self.frame_counter = 0

        self.debug_output_dir = None

        self.p_index = 0
        # self.w_step = 256
        # self.h_step = 384
        self.w_step = 128
        self.h_step = 128



    def encode_frame(self, frame, isIframe = False, no_index_referesh=False):

        """
        Input:
            frame: the PIL image
        Output:
            eframe: encoded frame, code is torch.tensor on GPU
            tot_size: the total size of p rame and I patch
        Note:
            this function will NOT update the reference
        """
        #print("steps:", self.h_step , self.w_step )
        self.frame_counter += 1
        frame = to_tensor(frame)
        if isIframe:
            code, shapex, shapey, size = bpg_encode(frame)
            dec_frame = bpg_decode(code, shapex, shapey)
            eframe = EncodedFrame(code, shapex, shapey, "I", self.frame_counter)
            return eframe, size, dec_frame
        else:
            assert self.reference_frame is not None
            # use p_index to compute which part to encode the I-frame
            c, h, w = frame.shape
            if w % self.w_step != 0 or h % self.h_step != 0:
                raise RuntimeError("w_step and h_step need to divide W and H")

            # encode P part
            code, dec_frame = self.ae_coder.encode(frame, self.reference_frame)
            names = ["res", "mv", "z"]
            codes = [code.get_tensor(n).quantized_code for n in names]
            dists = [code.get_tensor(n).distribution for n in names]
            psize = self.entropy_coder.calculate_sizes(codes, dists)

            eframe = EncodedFrame(code, None, None, "P", self.frame_counter)

            # encode I part
            w_tot = w / self.w_step
            h_tot = h / self.h_step
            w_offset = int((self.p_index % w_tot) * self.w_step)
            h_offset = int(((self.p_index // w_tot) % h_tot) * self.h_step)
            #print(f"P_index = {self.p_index}, w_offset = {w_offset}, h_offset = {h_offset}")
            part_iframe = frame[:, h_offset:h_offset+self.h_step, w_offset:w_offset+self.w_step]
            icode, shapex, shapey, isize = bpg_encode(part_iframe)
            # ed = time.perf_counter()
            # print("self.bpg_encode: ", (ed - st) * 1000)
            ipart = IPartFrame(icode, shapex, shapey, w_offset, h_offset)
            eframe.ipart = ipart
            eframe.isize = isize
            eframe.frame_type = "P"
            eframe.tot_size = psize + isize
            
            if no_index_referesh == False:
                self.p_index += 1
            # print(eframe.frame_type)
            return eframe, psize + isize, dec_frame

    def decode_frame(self, eframe:EncodedFrame):
        """
        Input:
            eframe: the encoded frame (EncodedFrame object)
        Output:
            frame: the decoded frame in torch.tensor (3,h,w) on GPU, which can be used as ref frame
        Note:
            this function will NOT update the reference
        """
        if eframe.frame_type == "I":
            out = bpg_decode(eframe.code, eframe.shapex, eframe.shapey)
            return out
        else:
            assert self.reference_frame is not None
            #out = self.ae_coder.decode(eframe.code, self.reference_frame, eframe.shapex, eframe.shapey)
            # st = time.perf_counter()
            out = self.ae_coder.decode(eframe.code, self.reference_frame)
            # torch.cuda.synchronize()
            # ed = time.perf_counter()
            # print("self.ae_coder.decode:", (ed - st) * 1000)
            if eframe.ipart is not None:
                ipart = eframe.ipart
                # st = time.perf_counter()
                idec = bpg_decode(ipart.code, ipart.shapex, ipart.shapey)
                # torch.cuda.synchronize()
                # ed = time.perf_counter()
                # print("self.bpg_decode:", (ed - st) * 1000)

                out[:, ipart.offset_height:ipart.offset_height+self.h_step, ipart.offset_width:ipart.offset_width+self.w_step] = idec
            
            return out


    def update_reference(self, ref_frame):
        """
        Input:
            ref_frame: reference frame in torch.tensor with size (3,h,w). On GPU
        """
        self.reference_frame = ref_frame

    def fit_frame(self, frame):
        """
        set the h_step and w_step for the encoder
        frame is a PIL image
        """
        w, h = frame.size
        self.h_step, self.w_step = set_hw_step(h, w)

    def get_avg_freeze_psnr(self, frames):
        res = []
        for idx, frame in enumerate(frames[2:]):
            img1 = to_tensor(frame)
            img2 = to_tensor(frames[idx-2])
            res.append(METRIC_FUNC(img1, img2))
        return float(np.mean(res))




def init_ae_model(qmap_quality=1):
    PRETRAINED_MODEL = "models/pretrained"
    models = {
            "128": AEModel(GraceAdapter({"path": f"{PRETRAINED_MODEL}/128.model"}, scale_factor=0.5)),
            "256": AEModel(GraceAdapter({"path": f"{PRETRAINED_MODEL}/256.model"}, scale_factor=0.5)),
            "512": AEModel(GraceAdapter({"path": f"{PRETRAINED_MODEL}/512.model"}, scale_factor=0.5)),
            "1024": AEModel(GraceAdapter({"path": f"{PRETRAINED_MODEL}/1024.model"}, scale_factor=0.5)),
            "2048": AEModel(GraceAdapter({"path": f"{PRETRAINED_MODEL}/2048.model"})),
            "4096": AEModel(GraceAdapter({"path": f"{PRETRAINED_MODEL}/4096_pretrain.model"})),
            }

    return models


def load_video(path, nframes=482):
    video_file = path
    global frames_origin, frames_decoded_sender, frames_decoded_receiver, codes, used_model_ids, w, h, shapex, shapey
    frames_origin = read_video_into_frames(video_file, nframes=nframes)
    k = 0
    # for frame in frames_origin:
    #     k+=1
    for i in range(0, nframes):
        codes.append(None)
        used_model_ids.append(None)
    frames_decoded_receiver = frames_origin[0]
    frames_decoded_sender = frames_origin[0]

def encode_frame(ae_model: AEModel, is_iframe, ref_frame, new_frame, no_index_referesh=False):
    """
    ref_frame: torch tensor C, H, W
    new_frame: PIL image

    returns:
        size in bytes 
        the eframe
    """
    if ref_frame is not None:
        ae_model.update_reference(ref_frame)
    else:
        if not is_iframe:
            raise RuntimeError("Cannot encode a P-frame without reference frame")

    eframe, size, dec_frame = ae_model.encode_frame(new_frame, is_iframe)
    return size, eframe, dec_frame

def decode_frame(ae_model: AEModel, eframe: EncodedFrame, ref_frame, loss):
    """
    ref_frame: the tensor frame in 3, h, w

    returns:
        decoded frame
    """
    if ref_frame is not None:
        ae_model.update_reference(ref_frame)
    else:
        if not eframe.frame_type == "I":
            raise RuntimeError("Cannot decode a P-frame without reference frame")

    if eframe.frame_type == "I":
        if loss > 0:
            print("Error! Cannot add loss on I frame, it will cause huge error!")
        decoded = ae_model.decode_frame(eframe)
        return decoded
    else:
        eframe.apply_loss(loss, 1)
        ae_model.update_reference(ref_frame)
        decoded = ae_model.decode_frame(eframe)
        return decoded

def encode_whole_video(frames, ae_model: AEModel):
    """
    Input:
        frames: a list of frames in PIL format
    Return:
        orig_frames: list of frames in torch.Tensor
        codes: list of EncodedFrame
        dec_frames: list of decoded frame in torch.Tensor
    """
    orig_frames = list(map(to_tensor, frames))
    codes = []
    dec_frames = []
    ref_frame = None
    for idx, frame in enumerate(frames):
        size, eframe, decoded_frame = encode_frame(ae_model, idx == 0, ref_frame, frame)
        eframe.tot_size = size
        #decoded_frame = decode_frame(ae_model, eframe, ref_frame, 0)
        codes.append(eframe)
        dec_frames.append(decoded_frame)
        ref_frame = decoded_frame
    return orig_frames, codes, dec_frames


def decode_with_loss(ae_model: AEModel, frame_id, losses, decoded_frames, eframes):
    """
    Input:
        frame_id: encode starting from xxx frame, should be larger than 1
        losses: list of loss values, the length determines how many frames will be decoded
        decoded_frames: the global decoded frames from encode_whole_video(), read-only
        eframes: the global eframes array from encode_whole_video(), read-only
    returns:
        damaged_frames: the list of damaged frames
    """
    damaged = []
    ref_frame = decoded_frames[frame_id - 1]
    for idx, loss in enumerate(losses):
        eframe = copy.deepcopy(eframes[frame_id + idx])
        damaged_frame = decode_frame(ae_model, eframe, ref_frame, loss)
        damaged.append(damaged_frame)
        ref_frame = damaged_frame
    return damaged




models = init_ae_model()

def run_one_model(model_id, input_pil_frames):
    total_frames_count = len(input_pil_frames)

    dfs = [] # size, psnr, ssim, loss, frame_id

    df = pd.DataFrame()
    model = models[model_id]
    model.p_index = 0
    orig_frames, codes, dec_frames = encode_whole_video(input_pil_frames, model)
    sizes = [code.tot_size for code in codes]
    psnrs = [PSNR(o, d) for o, d in zip(orig_frames, dec_frames)]
    ssims = [SSIM(o, d) for o, d in zip(orig_frames, dec_frames)]
    frame_ids = np.arange(0, total_frames_count)
    df["size"] = sizes
    df["psnr"] = psnrs
    df["ssim"] = ssims
    df["loss"] = 0
    df["frame_id"] = frame_ids
    df["nframes"] = 0
    #print(df)
    dfs.append(df)

    def run_multi_frame_losses(nframe, total_frames):
        dfs = []
        print("  - Running consecutive loss nframe =", nframe)
        for loss in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            damaged_frames = []
            df = pd.DataFrame()
            loss_arr = [loss] * nframe
            for frame_id in range(1, total_frames, nframe):
                damaged = decode_with_loss(model, frame_id, loss_arr, dec_frames, codes)
                damaged_frames.extend(damaged)
            df["size"] = [eframe.tot_size for eframe in codes[1:]]
            df["psnr"] = [PSNR(o, d) for o, d in zip(orig_frames[1:], damaged_frames)]
            df["ssim"] = [SSIM(o, d) for o, d in zip(orig_frames[1:], damaged_frames)]
            df["loss"] = loss
            df["frame_id"] = np.arange(1, total_frames)
            df["nframes"] = nframe
            dfs.append(df)
        return pd.concat(dfs)

    dfs += [run_multi_frame_losses(1, 16)]
    dfs += [run_multi_frame_losses(3, 16)]
    dfs += [run_multi_frame_losses(5, 16)]
    #run_multi_frame_losses(3, 16)
    #run_multi_frame_losses(5, 16)
    final_df = pd.concat(dfs)
    return final_df

def run_one_video(video):
    input_frames = read_video_into_frames(video, nframes=16)
    
    dfs = []
    for model_id in models.keys():
        print("  Running model:", model_id)
        df = run_one_model(model_id, input_frames)
        df["model_id"] = model_id
        dfs.append(df)

    final_df = pd.concat(dfs)
    return final_df

def run_one_file(index_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    videos = []
    with open(index_file, "r") as fin:
        for line in fin:
            videos.append(line.strip("\n"))

    video_dfs = []
    for idx, video in enumerate(videos):
        print(f"\033[33mRunning video: {video}, index: {idx}\033[0m")
        video_basename = os.path.basename(video)
        if os.path.exists(f"{output_dir}/{video_basename}.csv"):
            print(f"Skip the finished video: {video}")
            video_df = pd.read_csv(f"{output_dir}/{video_basename}.csv")
        else:
            video_df = run_one_video(video)
            video_df["video"] = video_basename
            video_df.to_csv(f"{output_dir}/{video_basename}.csv", index=None)
        video_dfs.append(video_df)

    final_df = pd.concat(video_dfs)
    final_df.to_csv(f"{output_dir}/all.csv", index=None)
    return final_df

run_one_file("INDEX.txt", "results/pretrained-debug")
