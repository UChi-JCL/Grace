import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

from multiprocessing import Pool
import torch.multiprocessing as mp
import multiprocessing
import pandas as pd
import os, sys
import subprocess as sp
import io
import shlex
from copy import deepcopy
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
import time
import ffmpeg

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import numpy as np
import cv2
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

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
    return float(ssim( Y1_raw.float().cuda().unsqueeze(0), Y1_com.float().cuda().unsqueeze(0), data_range=1, size_average=False).cpu().detach()) 

def metric_all_in_one(Y1_raw, Y1_com):
    """
    returns: 
        rgbpsnr, rgbssim, yuvpsnr, yuvssim
    """
    rgbpsnr = PSNR(Y1_raw, Y1_com)
    # breakpoint()
    rgbssim = float(ssim( Y1_raw.float().cuda().unsqueeze(0), Y1_com.float().cuda().unsqueeze(0), data_range=1, size_average=False).cpu().detach()) 

    # y1, u1, v1, yuv1 = RGB2YUV(Y1_raw, True)
    # y2, u2, v2, yuv2 = RGB2YUV(Y1_com, True)

    # yuvpsnr = PSNR_YUV(yuv1, yuv2)
    # yuvssim = SSIM_YUV(y1, y2)
    return float(rgbpsnr), rgbssim, 0, 0

METRIC_FUNC = metric_all_in_one 

def read_video_into_frames(video_path, frame_size=None, nframes=1000, temp_path=None):
    """
    Input:
        video_path: the path to the video
        frame_size: resize the frame to a (width, height), if None, it will not do resize
        nframes: number of frames
    Output:
        frames: a list of PIL images
    """
    def create_temp_path(temp_path = None):
        #process = multiprocessing.current_process()
        #pid = process.pid
        if temp_path is None:
            #path = f"/tmp/grace_frames-{pid}/"
            path = f"/tmp/grace_frames-{np.random.randint(0, 100000)}/"
        else:
            path = temp_path
        while os.path.isdir(path):
            print("Path:", path, "is existed! Change to a new one!")
            path = f"/tmp/grace_frames-{np.random.randint(0, 100000)}/"
        os.makedirs(path, exist_ok=True)
        print("Created temp path", path)
        return path

    def remove_temp_path(tmp_path):
        os.system("rm -rf {}".format(tmp_path))

    frame_path = create_temp_path(temp_path)
    if frame_size is None:
        cmd = f"ffmpeg -i {video_path} {frame_path}/%03d.png 2>/dev/null 1>/dev/null"
    else:
        width, height = frame_size
        cmd = f"ffmpeg -i {video_path} -s {width}x{height} {frame_path}/%03d.png 2>/dev/null 1>/dev/null"

    print(cmd)
    os.system(cmd)
    
    image_names = os.listdir(frame_path)
    frames = []
    for img_name in sorted(image_names)[:nframes]:
        frame = Image.open(os.path.join(frame_path, img_name))

        ''' pad to nearest 64 for DVC model '''
        padsz = 128
        w, h = frame.size
        pad_w = int(np.ceil(w / padsz) * padsz)
        pad_h = int(np.ceil(h / padsz) * padsz)
        if pad_h == 320:
            pad_h = 256 
        if pad_w == 320:
            pad_w = 256 
            if pad_h == 256:
                pad_w = 384

        if pad_w <= 192:
            pad_w *= 2
        if pad_h <= 192:
            pad_h *= 2

        frames.append(frame.resize((pad_w, pad_h)))

    remove_temp_path(frame_path)
    print("Resizing image to", frames[-1].size)
    return frames



class MPEGModel:
    def __init__(self, gop, only_P=True, use_codec="265"):
        self.qp = 15
        self.gop = gop
        self.only_P = only_P
        self.use_codec = use_codec

    def set_qp(self, q):
        self.qp = q

    def get_qp(self):
        return self.qp

    def get_avg_freeze_psnr(self, frames):
        res = []
        for idx, frame in tqdm(enumerate(frames[1:])):
            img1 = to_tensor(frame)
            img2 = to_tensor(frames[idx-1])
            res.append(METRIC_FUNC(img1, img2))
        v1, v2, v3, v4 = zip(*res)
        return np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4)


    def encode_video(self, frames, temp_outfile=None):
        """
        Input:
            frames: list of PIL images
        Output:
            psnrs: psnr for each frame
            bpp: average BPP of the encoded video
        """
        def get_tmp_outfile():
            output_filename = f'/tmp/output2-{np.random.randint(0, 10000000)}.mp4'
            while os.path.exists(output_filename):
                output_filename = f'/tmp/output2-{np.random.randint(0, 10000000)}.mp4'
            return output_filename

        def free_tmp_outfile(outfile):
            os.system("rm -f {}".format(outfile))

        assert self.qp is not None
        imgByteArr = io.BytesIO()
        fps = 25
        if temp_outfile is None:
            output_filename = get_tmp_outfile()
        else:
            output_filename = temp_outfile
        width, height = frames[0].size
        #cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -crf {self.qp} -g {self.gop} -sc_threshold 0 -loglevel debug {output_filename}'
        if self.use_codec == "264":
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -crf {self.qp} -g {self.gop} -sc_threshold 0 -loglevel debug {output_filename}'
        elif self.use_codec == "265":
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset veryfast -tune zerolatency -x265-params "crf={self.qp}:keyint={self.gop}:verbose=1" -sc_threshold 0 -loglevel debug {output_filename}'
        elif self.use_codec == "good-265":
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -tune zerolatency -x265-params "crf={self.qp}:keyint={self.gop}:verbose=1" -sc_threshold 0 -loglevel debug {output_filename}'
        elif self.use_codec == "vp9":
            est_bpp = np.power(2, (46 - self.qp) / 6) * 0.01
            bitrate_kbps = int(est_bpp * width * height * fps / 1000)
            print("estimated kbps = ", bitrate_kbps)
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -c:v libvpx-vp9 -b:v {bitrate_kbps}k -loglevel debug {output_filename}'
        else: 
            raise RuntimeError("Unknown codec name: " + self.use_codec)

        print(cmd)

        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        for img in tqdm(frames):
            process.stdin.write(np.array(img).tobytes())
        process.stdin.close()
        process.wait()
        process.terminate()

        if self.only_P:
            bpp_res = sp.check_output(f'ffprobe -show_frames {output_filename} | grep "pkt_size\|pict_type" | grep "=P" -B 1 | grep "pkt_size"',
                                            stderr=open("/dev/null","w"), shell=True, encoding="utf-8")
        else:
            bpp_res = sp.check_output(f"ffprobe -show_frames {output_filename} | grep pkt_size", stderr=open("/dev/null","w"), shell=True, encoding="utf-8")
        temp = bpp_res.split("\n")
        temp.remove("")
        sizes = list(map(lambda s: int(s.split("=")[1]), temp))
        bpps = list(map(lambda B: B * 8 / (height * width), sizes))

        # get psnr
        psnrs = []

        clip = read_video_into_frames(output_filename, temp_path=f"{output_filename}-dir")

        for i, frame in enumerate(frames):
            if self.only_P and i % self.gop == 0:
                continue
            Y1_raw = to_tensor(frame)
            Y1_com = to_tensor(clip[i])
            psnrs += [METRIC_FUNC(Y1_raw, Y1_com)]
        free_tmp_outfile(output_filename)
        print("Mean psnr and bpp:", np.mean([_[0] for _ in psnrs]), np.mean(bpps))
        return psnrs, bpps, sizes


def run_freeze_psnr_videos(index_file, nframes=10):
    with open(index_file, "r") as fin:
        videos = [l.strip("\n") for l in fin]

    def do_one_video(video_name):
        print("Running video:", video_name)
        frames = read_video_into_frames(video_name, None, nframes)
        dfs = []

        mpeg_model = MPEGModel(gop = 10)
        freeze_psnr = mpeg_model.get_avg_freeze_psnr(frames)
        df = pd.DataFrame()
        return video_name.split("/")[-1], *freeze_psnr

    res = [do_one_video(_)  for _ in videos]
    res = list(zip(*res))
    print("here: ", res)
    df = pd.DataFrame()
    df["video"] = res[0]
    df["psnr"] = res[1]
    df["ssim"] = res[2] 
    df["yuvpsnr"] = res[3] 
    df["yuvssim"] = res[4] 
    df["bpp"] = -1
    print(df)
    return df

def do_one_qp(qp, gop, use_codec, frames):
    mpeg_model = MPEGModel(gop = gop, use_codec = use_codec, only_P=False)
    mpeg_model.set_qp(qp)
    model_name = f"{use_codec}-{qp}" 
    temp_name = f"/tmp/grace2-{qp}-{gop}-{use_codec}.mp4"
    psnrs, bpps, sizes = mpeg_model.encode_video(frames, temp_name)
    ret = pd.DataFrame()
    rgbpsnrs, rgbssims, yuvpsnrs, yuvssims = zip(*psnrs)
    while len(bpps) < len(rgbpsnrs):
        bpps.append(bpps[-1])
    while len(sizes) < len(rgbpsnrs):
        sizes.append(sizes[-1])
    ret["psnr"] = rgbpsnrs
    ret["ssim"] = rgbssims
    ret["size"] = sizes
    ret["yuvpsnr"] = yuvpsnrs
    ret["yuvssim"] = yuvssims
    ret["bpp"] = bpps
    ret["loss"] = 0
    ret["frame_id"] = ret.index
    ret["model"] = model_name
    return ret

def run_one_video(video_name, use_codec = "265"):
    print("Running video:", video_name)
    nframes = 20
    gop = 100
    frames = read_video_into_frames(video_name, None, nframes)
    dfs = []
    
    mpeg_model = MPEGModel(gop = gop, use_codec = use_codec, only_P=False)
    
    qplist = [(qp, gop, use_codec, frames) for qp in range(9, 49, 3)]
    with mp.Pool(6) as p:
        dfs = p.starmap(do_one_qp, qplist)
    #dfs = [do_one_qp(*qp) for qp in qplist]

    df = pd.concat(dfs)
    df["video"] = video_name.split("/")[-1]
    return df

def run_one_file(index_file, output_dir, use_codec = "265"):
    """
    running MPEG, return a dataframe with:
    <video name> <frame_id> <loss = 0> <psnr> <bpp> <model name = "264/265 + qp">
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(index_file, "r") as fin:
        videos = [l.strip("\n") for l in fin]

    video_dfs = []
    for idx, video in enumerate(videos):
        print(f"\033[33mRunning video: {video}, index: {idx}\033[0m")
        video_basename = os.path.basename(video)
        if os.path.exists(f"{output_dir}/{video_basename}.csv"):
            print(f"Skip the finished video: {video}")
            video_df = pd.read_csv(f"{output_dir}/{video_basename}.csv")
        else:
            video_df = run_one_video(video, use_codec)
            video_df["video"] = video_basename
            video_df.to_csv(f"{output_dir}/{video_basename}.csv", index=None)
        video_dfs.append(video_df)

    final_df = pd.concat(video_dfs)
    final_df.to_csv(f"{output_dir}/all.csv", index=None)

    freeze_df = run_freeze_psnr_videos(index_file, 16)
    outfile = os.path.join(output_dir, f"freeze.csv")
    freeze_df.to_csv(outfile, index=None)
    return final_df

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn', force=True)
    
    run_one_file("INDEX_FINAL.txt", "results/h265", "265")
    run_one_file("INDEX_FINAL.txt", "results/h264", "264")
