import torch
import math
import time
import numpy as np
from .net import load_model, VideoCompressor
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
import torchac

class DVCModel:
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

    def encode(self, image, refer_frame, return_z = False, mask=None):
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
                mv, res = self.model.encode(image, refer_frame, return_z = False)
            else:
                mv, res, z = self.model.encode(image, refer_frame, return_z = True)
        shape_mv = mv.shape
        shape_res = res.shape
        code = torch.cat([torch.flatten(mv), torch.flatten(res)])
        # torch.cuda.synchronize()

        if not return_z:
            return code, shape_mv, shape_res
        else:
            return code, shape_mv, shape_res, z

    ''' custom z encoder by junchen '''
    def encode_z(self, code_res, shape_res):
        features = torch.reshape(code_res, shape_res)
        return self.model.respriorEncoder(features)

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
        if self.use_half:
            code = code.half()
            refer_frame = refer_frame.half()
        else:
            code = code.float()
            refer_frame = refer_frame.float()

        mvsize = np.prod(shape_mv)
        ressize = np.prod(shape_res)
        assert mvsize + ressize == torch.numel(code)

        refer_frame = refer_frame[None, :].to(self.device)
        code = code.to(self.device)
        mv = torch.reshape(code[:mvsize], shape_mv)
        res = torch.reshape(code[mvsize:], shape_res)

        with torch.no_grad():
            out = self.model.decode(refer_frame, mv, res)
        # torch.cuda.synchronize()
        # if self.first_time:
            # self.first_time = False
            # self.model.compile_individual()
        return torch.squeeze(out)

    def get_saliency(self, code, refer_frame, shape_mv, shape_res, orig_image):
        """
        Input:
            code: 1-D torch tensor contains mv and residual
            refer_frame: torch.tensor with shape 3,h,w
            shape_mv: shape of motion_vec
            shape_res: shape of residual
            orig_image: original image, torch.tensor in 3,h,w shape, for MSE computation
        Returns:
            saliency_mv: sailency for motion vector, in shape_mv
            saliency_res: sailency for residual, in shape_res
        """
        mvsize = np.prod(shape_mv)
        ressize = np.prod(shape_res)
        assert mvsize + ressize == torch.numel(code)

        refer_frame = refer_frame[None, :].to(self.device)
        code = code.to(self.device).type(torch.float)
        mv = torch.reshape(code[:mvsize], shape_mv)
        res = torch.reshape(code[mvsize:], shape_res)

        #import pdb
        #pdb.set_trace()

        ''' zero out the gradient '''
        self.model.zero_grad()

        ''' get output '''
        mv.requires_grad_(True)
        res.requires_grad_(True)
        refer_frame.requires_grad_(False)
        out = self.model.decode(refer_frame, mv, res)
        out = torch.squeeze(out)

        ''' compute loss '''
        mse_loss = torch.mean((out - orig_image).pow(2))
        mse_loss.backward()

        return torch.squeeze(mv.grad), torch.squeeze(res.grad)

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



class DVCEntropyCoder:
    def __init__(self, dvc_model: DVCModel):
        self.model = dvc_model.model
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
        for i in range(-mxrange, mxrange + 1):
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

    ''' custom size estimator by junchen '''
    def entropy_encode_with_size(self, code, mvsize, ressize, z, use_estimation = False):
        """
        Parameter:
            code: a 1-D torch tensor,
                  equals to torch.cat([mv.flatten(), residual.flatten()])
            size_mv: size of motion vec
            size_res: size of residual
        Returns:
            bytestream: it is None
            size: the size of the stream
        """
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
        #print(sz1, sz2, sz3)
        return None, sz1 + sz2 + sz3

    def entropy_decode(self):
        raise NotImplementedError

