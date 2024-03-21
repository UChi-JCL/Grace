## Academic Software License: © 2023 UChicago (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-4:
## 
## Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.
## Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.
## IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

import time
import torch
import numpy as np
from .net import load_model, VideoCompressor
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
import torchac

from .grace_model import GraceModel, GraceEntropyCoder


class GraceBasicCode:
    def __init__(self, code, shapex, shapey, z):
        self.code = code
        self.shapex = shapex
        self.shapey = shapey
        self.z = z
        self.ipart = None
        self.isize = 0

    def apply_loss(self, loss, no_use=None):
        leng = torch.numel(self.code)

        rnd = torch.rand(leng).to(self.code.device)
        rnd = (rnd > loss).long()
        rnd = rnd[:leng].reshape(self.code.shape)
        self.code = self.code * rnd

        if self.ipart is not None and np.random.random() < loss:
            self.ipart = None


class GraceInterface:
    def __init__(self, config, use_half=True, scale_factor=1):
        self.gracemodel = GraceModel(config, use_half, scale_factor)
        self.use_half = use_half

        if use_half:
            self.gracemodel.set_half()
        else:
            self.gracemodel.set_float()

        self.ecmodel = GraceEntropyCoder(self.gracemodel)


    def encode(self, image: torch.Tensor, refer_frame: torch.Tensor) -> GraceBasicCode:
        """
        Main interface of encode
        Input:
            image: torch.tensor with shape 3, h, w, fp32
            refer_frame: torch.tensor with shape 3, h, w, fp32
        Returns:
            code: the GraceBasicCode, can be used for decode and entropy encode
        """
        code, shapex, shapey, z = self.gracemodel.encode(image, refer_frame, return_z = True)
        return GraceBasicCode(code, shapex, shapey, z)

    def entropy_encode(self, code: GraceBasicCode):
        """
        Main interface of entropy encode
        Input:
            code: the output of function `encode()`
        Returns:
            a number that is the length of encoded bytestream
        """
        return self.ecmodel.entropy_encode(code.code, code.shapex, code.shapey, code.z, use_estimation=True)[1]

    def decode(self, code: GraceBasicCode, refer_frame: torch.Tensor) -> torch.Tensor:
        """
        Main interface of decode
        Input:
            code: the output of function `encode()`
            refer_frame: the image tensor with shape 3, h, w, fp32
        Returns:
            The decoded image tensor, shape (3, h, w), fp32
        """
        decoded = self.gracemodel.decode(code.code, refer_frame, code.shapex, code.shapey).float().clamp(0, 1)
        return decoded
