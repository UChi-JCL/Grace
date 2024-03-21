## Academic Software License: © 2023 UChicago (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-4:
## 
## Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.
## Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.
## IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

import torch
import math
import time
import numpy as np
from .net import load_model, VideoCompressor
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
import torchac

class GraceModel:
    """
    Fields in config:
        path: the path to the model
    """
    def __init__(self, config, use_half=False, scale_factor=1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "device" in config:
            self.device = config["device"]
        self.config = config
        self.model = VideoCompressor()

        load_model(self.model, config["path"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.use_half = use_half

        if self.use_half:
            self.model.half()

        self.model.set_scale_factor(scale_factor)
        self.first_time = True

    def set_half(self):
        self.model.half()
        self.model.respriorEncoder.float()
        self.model.respriorDecoder.float()
        self.use_half = True

    def set_float(self):
        self.model.float()
        self.use_half = False

    def encode_separate(self, image, refer_frame, return_z = False, return_dec = False):
        """
        Parameter:
            image: torch.tensor with shape 3,h,w, fp32
            refer_frame: torch.tensor with shape 3,h,w, fp32
        Returns:
            code: a 1-D torch tensor, encoded image representation, without EC
                  equals to torch.cat([mv.flatten(), residual.flatten()]),
                  when self.use_half == true, return fp16, else fp32
            shape_mv: shape of motion vec
            shape_res: shape of residual
        """

        if self.use_half:
            image = image.half()
            refer_frame = refer_frame.half()
        else:
            image = image.float()
            refer_frame = refer_frame.float()

        image = image[None, :].to(self.device)
        refer_frame = refer_frame[None, :].to(self.device)

        with torch.no_grad():
            if not return_z:
                mv, res, recon_frame = self.model.encode(image, refer_frame, return_z = False)
                z = None
            else:
                mv, res, z, recon_frame = self.model.encode(image, refer_frame, return_z = True)
        
        if return_dec:
            return mv, res, z, torch.squeeze(recon_frame)
        else:
            return mv, res, z

    def encode(self, image, refer_frame, return_z = False):
        mv, res, z = self.encode_separate(image, refer_frame, return_z, return_dec = False)
        shape_mv = mv.shape
        shape_res = res.shape
        code = torch.cat([torch.flatten(mv), torch.flatten(res)])

        if not return_z:
            return code, shape_mv, shape_res
        else:
            return code, shape_mv, shape_res, z

    def decode(self, code, refer_frame, shape_mv, shape_res):
        """
        Parameter:
            code: 1-D torch tensor contains mv and residual
            refer_frame: torch.tensor with shape 3,h,w
            shape_mv: shape of motion_vec
            shape_res: shape of residual
        Returns:
            image: torch.tensor with shape (3, h, w)
        """
        mvsize = np.prod(shape_mv)
        ressize = np.prod(shape_res)
        assert mvsize + ressize == torch.numel(code)

        code = code.to(self.device)
        mv = torch.reshape(code[:mvsize], shape_mv)
        res = torch.reshape(code[mvsize:], shape_res)

        return self.decode_separate(mv, res, refer_frame)

    def decode_separate(self, mv, res, refer_frame):
        """
        mv, res: shape is NCHW
        refer_frame: shape is CHW (no N)
        """
        if self.use_half:
            mv = mv.half()
            res = res.half()
            refer_frame = refer_frame.half()
        else:
            mv = mv.float()
            res = res.float()
            refer_frame = refer_frame.float()

        refer_frame = refer_frame[None, :].to(self.device)
        with torch.no_grad():
            out = self.model.decode(refer_frame, mv, res)
        return torch.squeeze(out)

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    Lp = cdf_float.shape[-1]
    PRECISION = 16
    factor = 2 ** PRECISION
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    val = cdf.cpu()
    return val

def encode_float_cdf_with_repeat(cdf_float, sym, repeats, needs_normalization=True, check_input_bounds=False):
    if check_input_bounds:
        if cdf_float.min() < 0:
            raise ValueError(f'cdf_float.min() == {cdf_float.min()}, should be >=0.!')
        if cdf_float.max() > 1:
            raise ValueError(f'cdf_float.max() == {cdf_float.max()}, should be <=1.!')
        Lp = cdf_float.shape[-1]
        if sym.max() >= Lp - 1:
            raise ValueError
    cdf_int = _convert_to_int_and_normalize(cdf_float, needs_normalization)
    cdf_int = cdf_int.repeat(repeats)
    return torchac.encode_int16_normalized_cdf(cdf_int, sym)



class GraceEntropyCoder:
    def __init__(self, grace_model: GraceModel):
        self.model = grace_model.model
        self.mv_cdfs = None
        self.mvshape = None
        self.z_cdfs = None
        self.zshape = None

    def cache_cdfs(self, x, bitest):
        n,c,h,w = x.shape
        cdfs = []
        for i in range(-self.model.mxrange, self.model.mxrange):
            #cdfs.append(bitest(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
            cdfs.append(bitest(i-0.5).view(1,c,1))
        cdfs = torch.cat(cdfs, 2).detach()
        return cdfs.view(1, c, 1, 1, -1)

    def getrealbits(self, x, cdfs):
        n,c,h,w = x.shape
        x = x + self.model.mxrange
        byte_stream = encode_float_cdf_with_repeat(cdfs, x.cpu().detach().to(torch.int16).clamp(max=self.model.mxrange*2-2), (1,1,h,w,1), check_input_bounds=False)
        size_in_bytes = len(byte_stream)
        return byte_stream, size_in_bytes


    def compress_res(self, res, sigma):
        """
        res: the residual torch tensor
        sigma: the sigma torch tensor, same size as residual
        """
        res = res.clamp(-16, 16)
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)

        cdfs = []
        mxrange = min(self.model.mxrange, torch.max(torch.abs(res)).item())
        mxrange = int(mxrange)
        res = res + mxrange #self.model.mxrange
        n,c,h,w = res.shape
        #for i in range(-self.model.mxrange, self.model.mxrange):
        for i in range(-mxrange, mxrange + 2):
            cdfs.append(gaussian.cdf(torch.tensor(i - 0.5)).view(n,c,h,w,1))


        cdfs = torch.cat(cdfs, 4).detach()
        tmp = res.cpu().detach().to(torch.int16)

        #byte_stream = torchac.encode_float_cdf(cdfs, res.cpu().detach().to(torch.int16), check_input_bounds=True)
        byte_stream = torchac.encode_float_cdf(cdfs, tmp, check_input_bounds=False)
        size_in_bytes = len(byte_stream)


        return byte_stream, size_in_bytes

    def compress_mv(self, mv, using_quant=False):
        if self.mvshape is None or self.mvshape != mv.shape:
            self.mvshape = mv.shape
            self.mv_cdfs = self.cache_cdfs(mv, self.model.bitEstimator_mv)

        #bs, size = self.getrealbits(mv, self.model.bitEstimator_mv)
        bs, size = self.getrealbits(mv, self.mv_cdfs)

        return bs, size

    def compress_z(self, z):
        if self.zshape is None or self.zshape != z.shape:
            self.zshape = z.shape
            self.z_cdfs = self.cache_cdfs(z, self.model.bitEstimator_z)
        #bs, size = self.getrealbits(z, self.model.bitEstimator_z)
        bs, size = self.getrealbits(z, self.z_cdfs)
        return bs, size

    def estimate_res(self, res, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(res + 0.5) - gaussian.cdf(res - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        total_bits = total_bits.item()
        return None, total_bits / 8

    def estimate_mv(self, mv):
        prob = self.model.bitEstimator_mv(mv + 0.5) - self.model.bitEstimator_mv(mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return None, total_bits.item() / 8

    def estimate_z(self, z):
        prob = self.model.bitEstimator_z(z + 0.5) - self.model.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return None, total_bits.item() / 8


    def entropy_encode(self, code, shape_mv, shape_res, z, use_estimation = False):
        """
        Parameter:
            code: a 1-D torch tensor,
                  equals to torch.cat([mv.flatten(), residual.flatten()])
            shape_mv: shape of motion vec
            shape_res: shape of residual
        Returns:
            bytestream: it is None
            size: the size of the stream
        """
        code = code.float()
        z = z.float()

        mvsize = np.prod(shape_mv)
        ressize = np.prod(shape_res)
        assert mvsize + ressize == torch.numel(code)


        mv = torch.reshape(code[:mvsize], shape_mv)
        res = torch.reshape(code[mvsize:], shape_res)
        sigma = self.model.respriorDecoder(z)

        if use_estimation:
            bs1, sz1 = self.estimate_res(res, sigma)
            bs2, sz2 = self.estimate_mv(mv)
            bs3, sz3 = self.estimate_z(z)
        else:
            bs1, sz1 = self.compress_res(res, sigma)
            bs2, sz2 = self.compress_mv(mv)
            bs3, sz3 = self.compress_z(z)
        return None, sz1 + sz2 + sz3

    def entropy_decode(self):
        raise NotImplementedError

