import os
import yaml
import time
import struct
import shutil
import numpy as np
from shutil import copy2
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def apply_loss(code, loss_rate, blocksize = 100):
    leng = len(code)
    nblocks = leng // blocksize 
    #if nblocks * loss_rate < 1:
    #    print("Warning: Number of blocks ({}) are very small! Loss rate = {}".format(nblocks, loss_rate))

    newcode = bytearray()
    for i in range(0, leng, blocksize):
        dice = np.random.uniform(0, 1)
        codelen = min(i + blocksize, leng) - i
        if dice <= loss_rate:
            newcode = newcode + bytearray(b"\x00") * codelen
        else:
            newcode = newcode + code[i:i+codelen]
    return newcode

def apply_loss_numpy(code, loss_rate, blocksize):
    sz = (code.size // blocksize) + 1
    nzeros = int(sz * loss_rate)
    indice = np.random.choice(np.arange(sz), replace = False, size=nzeros)
    mask = np.ones(sz)
    mask[indice] = 0
    mask = np.repeat(mask, blocksize)[:code.size]
    #print(code.size, mask.size, mask == 0)

    newcode = code.copy()
    np.putmask(newcode, mask == 0, np.zeros(newcode.shape, dtype = newcode.dtype))
    return newcode



def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        return config


def init(args):
    base_dir = f'./results/{args.name}'
    snapshot_dir = os.path.join(base_dir, 'snapshots')
    output_dir = os.path.join(base_dir, 'outputs')
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    config = get_config(args.config)
    try:
        copy2(args.config, os.path.join(base_dir, 'config.yaml'))
    except shutil.SameFileError:
        pass

    return config, base_dir, snapshot_dir, output_dir, log_dir


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, config, base_dir, snapshot_dir, output_dir, log_dir, level_num=11, only_print=False):
        self.config = config
        self.base_dir = base_dir
        self.snapshot_dir = snapshot_dir
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.level_num = level_num
        self.itr = 0
        self.init()

        if not only_print:
            self._init_summary_writers(level_num)

    def _init_summary_writers(self, level_num):
        self.writer = SummaryWriter(self.log_dir)
        self.test_writers = [SummaryWriter(os.path.join(self.log_dir, f'level_{i}')) for i in range(level_num + 1)]

    def init(self):
        self.loss = AverageMeter()
        self.bpp_loss = AverageMeter()
        self.mse_loss = AverageMeter()
        self.psnr = AverageMeter()
        self.ms_ssim = AverageMeter()
        self.aux_loss = AverageMeter()

    def load_itr(self, itr):
        self.itr = itr

    def update(self, out_criterion, aux_loss):
        self.loss.update(out_criterion['loss'].item())
        self.bpp_loss.update(out_criterion['bpp_loss'].item())
        self.mse_loss.update(out_criterion['mse_loss'].item())
        self.aux_loss.update(aux_loss.item())
        self.itr += 1

    def update_test(self, bpp, psnr, ms_ssim, out_criterion, aux_loss):
        self.loss.update(out_criterion['loss'].item())
        self.bpp_loss.update(bpp.item())
        self.mse_loss.update(out_criterion['mse_loss'].item())
        self.psnr.update(psnr.item())
        self.ms_ssim.update(ms_ssim.item())
        self.aux_loss.update(aux_loss.item())

    def print(self, logfile=None):
        print(
            f'[{self.itr:>7}]'
            f' Total: {self.loss.avg:.4f} |'
            f' BPP: {self.bpp_loss.avg:.4f} |'
            f' MSE: {self.mse_loss.avg:.6f} |'
            f' Aux: {self.aux_loss.avg:.0f}'
        )
        if logfile:
            print(
                f'[{self.itr:>7}]'
                f' Total: {self.loss.avg:.4f} |'
                f' BPP: {self.bpp_loss.avg:.4f} |'
                f' MSE: {self.mse_loss.avg:.6f} |'
                f' Aux: {self.aux_loss.avg:.0f}',
                file=logfile
            )

    def print_test(self, case=-1, logfile=None):
        print(
            f'[ Test{case:>2} ]'
            f' Total: {self.loss.avg:.4f} |'
            f' BPP: {self.bpp_loss.avg:.4f} |'
            f' PSNR: {self.psnr.avg:.4f} |'
            f' MS-SSIM: {self.ms_ssim.avg:.4f} |'
            f' Aux: {self.aux_loss.avg:.0f}'
        )
        if logfile:
            print(
                f'[ Test{case:>2} ]'
                f' Total: {self.loss.avg:.4f} |'
                f' BPP: {self.bpp_loss.avg:.4f} |'
                f' PSNR: {self.psnr.avg:.4f} |'
                f' MS-SSIM: {self.ms_ssim.avg:.4f} |'
                f' Aux: {self.aux_loss.avg:.0f}',
                file=logfile
            )

    def write(self):
        self.writer.add_scalar('Total loss', self.loss.avg, self.itr)
        self.writer.add_scalar('BPP loss', self.bpp_loss.avg, self.itr)
        self.writer.add_scalar('MSE loss', self.mse_loss.avg, self.itr)
        self.writer.add_scalar('Aux loss', self.aux_loss.avg, self.itr)

    def write_test(self, level=0):
        if self.level_num == 1:
            writer = self.writer
        else:
            writer = self.test_writers[level]
        writer.add_scalar('[Test] Total loss', self.loss.avg, self.itr)
        writer.add_scalar('[Test] BPP', self.bpp_loss.avg, self.itr)
        writer.add_scalar('[Test] MSE loss', self.mse_loss.avg, self.itr)
        writer.add_scalar('[Test] PSNR', self.psnr.avg, self.itr)
        writer.add_scalar('[Test] MS-SSIM', self.ms_ssim.avg, self.itr)
        writer.add_scalar('[Test] Aux loss', self.aux_loss.avg, self.itr)


def save_checkpoint(filename, itr, model, optimizer, aux_optimizer, scaler=None):
    snapshot = {
        'itr': itr,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'aux_optimizer': aux_optimizer.state_dict()
    }
    if scaler is not None:
        snapshot['scaler'] = scaler.state_dict()
    torch.save(snapshot, filename)


def load_checkpoint(path, model, optimizer=None, aux_optimizer=None, scaler=None, only_net=False):
    snapshot = torch.load(path)
    itr = snapshot['itr']
    print(f'Loaded from {itr} iterations')
    model.load_state_dict(snapshot['model'])
    if not only_net:
        if 'optimizer' in snapshot:
            optimizer.load_state_dict(snapshot['optimizer'])
        if 'aux_optimizer' in snapshot:
            aux_optimizer.load_state_dict(snapshot['aux_optimizer'])
        if scaler is not None and 'scaler' in snapshot:
            scaler.load_state_dict(snapshot['scaler'])

    return itr, model


###############################################################################
import compressai

metric_ids = {
    "mse": 0,
}


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return 0, 0  # model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        model_id,  # inverse_dict(model_ids)[model_id],
        metric,  # inverse_dict(metric_ids)[metric],
        quality,
    )


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )

def save_raw_code(y, z, original_size, output):
    y_s = y.cpu().numpy().astype(np.int16)
    z_s = z.cpu().numpy().astype(np.int16)
    #print(y_s.size, z_s.size)
    #print(np.max(y_s), np.min(y_s))
    y_s1 = apply_loss_numpy(y_s, 0.0, 1)
    z_s1 = apply_loss_numpy(z_s, 0.0, 1)
    #assert not np.array_equal(z_s, z_s1)
    #assert not np.array_equal(y_s, y_s1)
    original_size = np.asarray(original_size)
    np.savez(output, y=y_s1, z=z_s1, os=original_size)
    #print(y.shape, z.shape)
    #print(len(z_s.reshape(-1).tobytes()) + len(y_s.reshape(-1).tobytes()))

def load_raw_numpy_code(infile):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not infile.endswith(".npz"):
        infile = infile + ".npz"

    npzfile = np.load(infile, allow_pickle=True)
    y, z, os = npzfile["y"], npzfile["z"], npzfile["os"]
    y = torch.from_numpy(y).to(device)
    z = torch.from_numpy(z).to(device)
    return y, z, (os[0], os[1])


def save_compressed_code(out, header, h, w, output):
    shape = out["shape"]
    with Path(output).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h, w))
        # write shape and number of encoded latents
        write_uints(f, (shape[0], shape[1], len(out["strings"])))
        for s in out["strings"]:
            new_s = apply_loss(s[0], 0.00, 50)
            assert len(new_s) == len(s[0])
            #print(len(s[0]), len(new_s))
            write_uints(f, (len(s[0]),))
            write_bytes(f, new_s)

def load_compressed_code(inputpath):
    with Path(inputpath).open("rb") as f:
        model_, metric, quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(f, 2)
        shape = read_uints(f, 2)
        strings = []
        n_strings = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            strings.append([s])

    return strings, shape, original_size


def _encode(model, x: torch.Tensor, output: str, qmap=None, metric='mse', coder='ans', quality=1, verbose=False, compress=True):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    start = time.time()
    net = model
    load_time = time.time() - start

    _, _, h, w = x.shape
    p = 64
    x = pad(x, p)

    with torch.no_grad():
        if qmap is None:
            if compress:
                out = net.compress(x)
            else:
                out = net.encode(x)
        else:
            if compress:
                out = net.compress(x, qmap)
            else:
                out = net.encode(x, qmap)

    shape = out["shape"]
    header = get_header(model, metric, quality)

    # save the data to file
    if compress:
        save_compressed_code(out, header, h, w, output)
    else:
        save_raw_code(out["strings"][0], out["strings"][1], [h, w], output)

    enc_time = time.time() - enc_start
    size = filesize(output)
    bpp = float(size) * 8 / (h * w)
    if verbose:
        print(
            f"{bpp:.4f} bpp |"
            f" Encoded in {enc_time:.4f}s (model loading: {load_time:.4f}s)"
        )
    return bpp, out, enc_time


def _decode(model, inputpath, coder='ans', verbose=False, compress=True):
    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    if compress:
        strings, shape, original_size = load_compressed_code(inputpath)
    else:
        y, z, original_size = load_raw_numpy_code(inputpath)
        strings = [y, z]

    start = time.time()
    net = model
    load_time = time.time() - start

    with torch.no_grad():
        if compress:
            out = net.decompress(strings, shape)
        else:
            out = net.decode(strings)

    x_hat = crop(out["x_hat"], original_size)
    x_hat.clamp_(0, 1)
    dec_time = time.time() - dec_start
    if verbose:
        print(f"Decoded in {dec_time:.4f}s (model loading: {load_time:.4f}s)")

    return x_hat, dec_time
