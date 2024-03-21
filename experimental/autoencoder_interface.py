## Academic Software License: © 2023 UChicago (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-4:
## 
## Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.
## Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.
## IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

import torch
from typing import Tuple, Dict
from enum import Enum

from grace.grace_model import GraceModel

class PacketizationMethod(Enum):
    RANDOMIZED = 0
    REPLICATED = 1

class TensorContext:
    def __init__(self,
            quantized_code: torch.Tensor,
            mxrange: int,
            distribution: torch.Tensor,
            packetization_method: PacketizationMethod):

        self.quantized_code = quantized_code
        self.mxrange = mxrange
        self.distribution = distribution
        self.packetization_method = packetization_method

    def simulate_loss(self, loss_ratio, blocksize = 1):
        leng = torch.numel(self.quantized_code)
        nblocks = (leng - 1) // blocksize + 1

        rnd = torch.rand(nblocks).to(self.quantized_code.device)
        rnd = (rnd > loss_ratio).long()
        #print("DEBUG: loss ratio =", loss_ratio, ", first 16 elem:", rnd[:16])
        rnd = rnd.repeat_interleave(blocksize)
        rnd = rnd[:leng].reshape(self.quantized_code.shape)
        self.quantized_code = self.quantized_code * rnd


class AECode:
    def __init__(self):
        self.codes = {}
        pass

    def add_tensor(self, 
            name: str, 
            quantized_code: torch.Tensor, 
            mxrange: int,
            distribution: torch.Tensor, 
            packetization_method: PacketizationMethod):
        """
        Input: 
            name: the name of the tensor to be added
            code: the quantized tensor of shape N, C, H, W, value range should be 0~L-1
            mxrange: the shift-range during quantization
            distribution: the distribution of the code, shape should be N, C, H, W, L + 1
            packetization_method: the enum class PacketizationMethod, tells packetizer how to packetize the code
        Note:
            distribution could be None at the receiver side
        """
        self.codes[name] = TensorContext(quantized_code, mxrange, distribution, packetization_method)

    def get_tensor(self, name: str) -> TensorContext:
        if name not in self.codes:
            raise RuntimeError("There is no code named " + name + " in the ae code")
        return self.codes[name]


class AEAdapter:
    """
    AEAdapter acts as a adapter between autoencoder models and packetizer/entropy codecs
    """

    def __init__(self):
        pass

    def encode(self, image: torch.Tensor, refer_frame: torch.Tensor) -> AECode:
        """
        Main interface of encode
        Input:
            image: torch.tensor with shape 3, h, w, fp32
            refer_frame: torch.tensor with shape 3, h, w, fp32
        Returns:
            code: the AECode, can be used for decode and entropy encode
            decoded_img: the decoded image which will be used as reference image for the next frame
        """
        pass

    def decode(self, aecode: AECode, refer_frame) -> torch.Tensor:
        """
        Main interface of decode
        Input:
            code: the output of function `encode()`
            refer_frame: the image tensor with shape 3, h, w, fp32
        Returns:
            The decoded image tensor, shape (3, h, w), fp32
        """
        pass

    def get_available_distribution(self, partial_code: AECode) -> Dict[str, torch.Tensor]:
        """
        Try to get the distributions based on the currently entropy-decoded tensors
        Input:
            partial_code: AECde, where the undecoded tensor and unknown distributions will be None
        Returns:
            a dictionary of name ->  tensor, where 
        """
        pass


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    Lp = cdf_float.shape[-1]
    PRECISION = 16
    factor = 2 ** PRECISION
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - Lp
    cdf_float = cdf_float.float().mul(new_max_value)
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    val = cdf.cpu()
    return val

class GraceAdapter(AEAdapter):
    def __init__(self, config, use_half=True, scale_factor = 1):
        super().__init__()
        self.gracemodel = GraceModel(config, use_half, scale_factor)
        self.use_half = use_half

        if use_half:
            self.gracemodel.set_half()
        else:
            self.gracemodel.set_float()

        self.cached_mv_dist = None
        self.cached_z_dist = None
        self.resshape = None
        self.mvshape = None
        self.zshape = None

    def _cache_cdfs(self, x, bitest):
        """
        returns the cdf of shape N,C,H,W,L
        """
        n,c,h,w = x.shape
        cdfs = []
        for i in range(-self.gracemodel.model.mxrange, self.gracemodel.model.mxrange+2):
            cdfs.append(bitest(i-0.5).view(1,c,1))
        cdfs = torch.cat(cdfs, 2).detach()
        cdfs_int = _convert_to_int_and_normalize(cdfs, True)
        return cdfs_int.view(1, c, 1, 1, -1).repeat(1, 1, h, w, 1)

    def _get_res_cdf(self, mxrange, resshape, sigma):
        """
        returns cdf 
        """
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)

        cdfs = []
        n,c,h,w = resshape

        for i in range(-mxrange, mxrange + 2):
            cdfs.append(gaussian.cdf(torch.tensor(i - 0.5)).view(n,c,h,w,1))

        cdfs = torch.cat(cdfs, 4).detach()
        return _convert_to_int_and_normalize(cdfs, True)


    def encode(self, image: torch.Tensor, refer_frame: torch.Tensor) -> Tuple[AECode, torch.Tensor]:
        """
        Main interface of encode
        Input:
            image: torch.tensor with shape 3, h, w, fp32
            refer_frame: torch.tensor with shape 3, h, w, fp32
        Returns:
            code: the AECode, can be used for decode and entropy encode
            decoded_img: the decoded image which will be used as reference image for the next frame
        """
        mv, res, z, dec_frame = self.gracemodel.encode_separate(image, refer_frame, True, True)
        code = AECode()

        ''' do mv '''
        mxrange_mv = self.gracemodel.model.mxrange
        mv = (mv + mxrange_mv).clamp(0, mxrange_mv * 2)
        if self.cached_mv_dist is None or self.mvshape != mv.shape:
            self.mvshape = mv.shape
            self.cached_mv_dist = self._cache_cdfs(mv, self.gracemodel.model.bitEstimator_mv)

        code.add_tensor("mv", mv, mxrange_mv, self.cached_mv_dist, PacketizationMethod.RANDOMIZED)

        ''' do z '''
        mxrange_z = self.gracemodel.model.mxrange
        z = (z + mxrange_z).clamp(0, mxrange_z * 2)
        if self.cached_z_dist is None or self.zshape != z.shape:
            self.zshape = z.shape
            self.cached_z_dist = self._cache_cdfs(z, self.gracemodel.model.bitEstimator_z)

        code.add_tensor("z", z, mxrange_z, self.cached_z_dist, PacketizationMethod.REPLICATED)

        ''' do res '''
        with torch.no_grad():
            sigma = self.gracemodel.model.respriorDecoder(z - mxrange_z)
        res = res.clamp(-16, 16)
        mxrange_res = min(self.gracemodel.model.mxrange, torch.max(torch.abs(res)).item())
        mxrange_res = int(mxrange_res)
        res = res + mxrange_res
        cdfs = self._get_res_cdf(mxrange_res, res.shape, sigma)
        self.resshape = res.shape

        code.add_tensor("res", res, mxrange_res, cdfs, PacketizationMethod.RANDOMIZED)

        return code, dec_frame

    def decode(self, aecode: AECode, refer_frame: torch.Tensor) -> torch.Tensor:
        """
        Main interface of decode
        Input:
            code: the output of function `encode()`
            refer_frame: the image tensor with shape 3, h, w, fp32
        Returns:
            The decoded image tensor, shape (3, h, w), fp32
        """
        mv_context = aecode.get_tensor("mv")
        res_context = aecode.get_tensor("res")

        mv = mv_context.quantized_code - mv_context.mxrange
        res = res_context.quantized_code - res_context.mxrange


        return self.gracemodel.decode_separate(mv, res, refer_frame)

    def get_available_distribution(self, partial_code: AECode) -> Dict[str, torch.Tensor]:
        """
        Try to get the distributions based on the currently entropy-decoded tensors
        Input:
            partial_code: AECde, where the undecoded tensor and unknown distributions will be None
        Returns:
            a dictionary of name ->  tensor, where 
        """
        ret = {}

        mv_ctx = partial_code.get_tensor("mv")
        res_ctx = partial_code.get_tensor("res")
        z_ctx = partial_code.get_tensor("z")

        ''' mv '''
        if mv_ctx.distribution is None:
            if self.cached_mv_dist is None:
                raise RuntimeError("We don't know the size of the frame yet, please call 'encode()' with fake frames first!")
            ret["mv"] = self.cached_mv_dist

        ''' z '''
        if z_ctx.distribution is None:
            if self.cached_z_dist is None:
                raise RuntimeError("We don't know the size of the frame yet, please call 'encode()' with fake frames first!")
            ret["z"] = self.cached_z_dist

        ''' res '''
        if res_ctx.distribution is None and z_ctx.quantized_code is not None:
            if self.resshape is None:
                raise RuntimeError("We don't know the size of the frame yet, please call 'encode()' with fake frames first!")
            z = z_ctx.quantized_code - z_ctx.mxrange
            with torch.no_grad():
                sigma = self.gracemodel.model.respriorDecoder(z.cuda())
            ret["res"] = self._get_res_cdf(res_ctx.mxrange, self.resshape, sigma)

        return ret

if __name__ == "__main__":
    adapter = GraceAdapter({"path": f"models/grace/4096_freeze.model"}, True, 1)

    frame1 = torch.zeros((3, 768, 1280))
    frame2 = torch.full((3, 768, 1280), 0.1)

    code = adapter.encode(frame2, frame1)

    dec_frame = adapter.decode(code, frame1)

    from copy import deepcopy

    code_copy = deepcopy(code)
    #code_copy.codes["mv"].quantized_code = None
    #code_copy.codes["mv"].distribution = None
    code_copy.codes["res"].quantized_code = None
    code_copy.codes["res"].distribution = None
    #code_copy.codes["z"].quantized_code = None
    #code_copy.codes["z"].distribution = None

    res = adapter.get_available_distribution(code_copy)
    print(list(res.keys()))

    import streaming_interface
    ec = streaming_interface.EntropyCodec()
    print(ec.calculate_size(code.get_tensor("mv").quantized_code, code.get_tensor("mv").distribution))
