## Academic Software License: © 2023 UChicago (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-4:
## 
## Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.
## Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.
## IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

import torchac
import math
from typing import List
import torch

def compress_res_faster(res, cdf, mxrange):
    """
    cdf is an array of (mxrange * 2 + 2) elements
    """
    cdfs = cdf.repeat(res.shape + (1,))
    res = res + mxrange
    tmp = res.cpu().detach().to(torch.int16)
    byte_stream = torchac.encode_float_cdf(cdfs, tmp, check_input_bounds=False)
    return byte_stream, len(byte_stream)

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

class PerPacketResEC:
    def __init__(self):
        NSIGMA = 64
        self.sigmas = list(map(lambda v: (1.05 ** v) * 0.03, np.arange(0, NSIGMA, 1)))
        self.shape = 3840
        self.mxrange = 7
        self.dists = torch.distributions.laplace.Laplace(0, torch.tensor(self.sigmas))
        self.P = []
        self.cdfs = []
        for i in range(-self.mxrange, self.mxrange + 2):
            self.P.append((self.dists.cdf(torch.tensor(i+0.5)) - self.dists.cdf(torch.tensor(i-0.5))).view(NSIGMA, 1))
            self.cdfs.append((self.dists.cdf(torch.tensor(i-0.5))).view(NSIGMA, 1))
        self.P = torch.cat(self.P, 1).cuda()
        self.cdfs = torch.cat(self.cdfs, 1).cuda()

    def compress_channel(self, res_channel):
        x = res_channel.flatten().long() + self.mxrange
        idx = torch.sum(-torch.log(self.P[:, x] + 1e-5), axis = 1).argmin().item()
        #bs, size = compress_res(x.reshape(1,1,48,80)-self.mxrange, torch.full((1, 1, 48, 80), self.sigmas[idx]), self.mxrange)
        bs, size = 0, 0 #compress_res_faster(res_channel, self.cdfs[idx], self.mxrange)
        return bs, size, idx

    def compress_channels(self, res):
        sizes = []
        indexes = []
        for channel in range(len(res)):
            bs, size, idx = self.compress_channel(res[channel])
            sizes.append(size)
            indexes.append(idx)
        return sizes, indexes

    def compress_channels_faster(self, res):
        x = (res.view(96, -1) + self.mxrange).long()
        indexes = torch.sum(-torch.log(self.P[:, x] + 1e-5), axis = 2).argmin(axis=0)
        c, h, w = res.shape
        cdfs = self.cdfs[indexes, :].view(96, 1, -1)
        cdf_int = _convert_to_int_and_normalize(cdfs, True)
        cdf_int = cdf_int.repeat((1, h*w, 1)).to(torch.int16)
        tmp = x.cpu().to(torch.int16)
        byte_stream = torchac.encode_int16_normalized_cdf(cdf_int, tmp)
        return len(byte_stream), indexes


class EntropyCodec:
    def __init__(self):
        pass


    def entropy_encode(self, tensor: torch.Tensor, distribution: torch.Tensor):
        """
        Input:
            tensor: the input tensor, should be quantized and shifted 
            distribution: the distribution of the tensor, should be int16 CPU
        Output:
            bs: the bytestream
            size: size of the bytestream in bytes
        """
        bs = torchac.encode_int16_normalized_cdf(distribution, tensor.cpu().to(torch.int16))

        return bs, len(bs)

    def entropy_decode(self, bs, distribution: torch.Tensor):
        """
        Input:
            bs: the encoded bytestream
            distribution: the distribution for decode, should be int16 CPU
        Output:
            the decoded tensor in torch.int16
        """
        res = torchac.decode_int16_normalized_cdf(distribution, bs)

        return res

    def calculate_size(self, tensor: torch.Tensor, distribution: torch.Tensor):
        """
        Input:
            tensor: the input tensor
            distribution: the distribution of the tensor, should be int16 CPU
        Output:
            size: the size calculated based on the entropy
        """
        tensor = tensor.long().cpu()
        N, C, H, W, L = distribution.shape
        index_n, index_c, index_h, index_w = torch.meshgrid(torch.arange(N), torch.arange(C), torch.arange(H), torch.arange(W))
        final_indices = (index_n, index_c, index_h, index_w, tensor)
        final_indices_p1 = (index_n, index_c, index_h, index_w, tensor + 1)
        probs = distribution[final_indices_p1] - distribution[final_indices]
        probs = (probs.float() + (probs < 0).long() * 65536.0) / 65536.0

        total_bits = torch.sum(torch.clamp(-1 * torch.log(probs + 1e-6) / math.log(2), 0, 50))
        return total_bits / 8

    def calculate_sizes(self, tensors: List[torch.Tensor], distributions: List[torch.Tensor]):
        total = 0
        for t, d in zip(tensors, distributions):
            total += self.calculate_size(t, d)
        return total

