import torch
import numpy as np
import qmap.models.models as models
import qmap.models.models_4x as models4x
import time

from .utils import load_checkpoint


class QmapModel:
    """
    To load the QmapModel, the following configuration is required:
        path: the model path
        name: the name of the model, use "available_models" to get the available model names
        N, M, sft_ks: the model parameters
        quality: value in the qmap
    It will load the model to GPU by default
    """
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "device" in config:
            self.device = config["device"]
        model_class = QmapModel.GetModel(config["name"])
        self.model = model_class(N=config["N"], M=config["M"], sft_ks=config["sft_ks"], prior_nc=64)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.config = config
        if self.device == "cpu":
            print("loading to cpu")
            snapshot = torch.load(config["path"], map_location=torch.device("cpu"))
        else:
            snapshot = torch.load(config["path"])
        state_dict = snapshot["model"]
        self.model.load_state_dict(state_dict)
        self.model.update()

    def encode(self, image: torch.Tensor):
        """
        Parameters:
            image: torch.tensor with shape (3, h, w)
        Returns:
            code: a 1-D torch.tensor, encoded image representation, WITHOUT entropy encoding!
            shapex: the original shape for the first output tensor
            shapey: the original shape for the second output tensor
        """

        image = image[None, :]
        qmap_shape = (image.size()[0], 1, *image.size()[-2:])
        qmap = torch.zeros(qmap_shape)
        qmap[:] = self.config["quality"]
        image = image.to(self.device)
        qmap = qmap.to(self.device)
        

        with torch.no_grad():
            out = self.model.encode(image, qmap)

            out2 = self.model.compress(image, qmap)

        shapex = out["strings"][0].shape
        shapey = out["strings"][1].shape
        x = torch.flatten(out["strings"][0])
        y = torch.flatten(out["strings"][1])
        code = torch.cat([x, y])

        size = len(out2["strings"][0][0]) + len(out2["strings"][1][0])
        torch.cuda.synchronize()
        return code, shapex, shapey, size

    def compress(self, image: torch.Tensor):
        """
        Parameters:
            image: torch.tensor with shape (3, h, w)
        Returns:
            code: a 1-D torch.tensor, encoded image representation, WITHOUT entropy encoding!
            shapex: the original shape for the first output tensor
            shapey: the original shape for the second output tensor
        """
        pass


    def encode_n(self, image: torch.Tensor):
        """
        Parameters:
            image: torch.tensor with shape (N, 3, h, w)
        Returns:
            code: a 1-D torch.tensor, encoded image representation, WITHOUT entropy encoding!
            shapex: the original shape for the first output tensor
            shapey: the original shape for the second output tensor
        """
        qmap_shape = (image.size()[0], 1, *image.size()[-2:])
        qmap = torch.zeros(qmap_shape)
        qmap[:] = self.config["quality"]
        image = image.to(self.device)
        qmap = qmap.to(self.device)

        with torch.no_grad():
            out = self.model.encode(image, qmap)
        shapex = out["strings"][0].shape
        shapey = out["strings"][1].shape
        x = torch.flatten(out["strings"][0])
        y = torch.flatten(out["strings"][1])
        code = torch.cat([x, y])

        return code, shapex, shapey

    def decode(self, code: torch.Tensor, shapex, shapey):
        """
        Parameters:
            code: a 1-D torch.tensor, encoded image representation, WITHOUT entropy encoding
        Returns:
            image: torch.tensor with shape (3, h, w)
        """
        xsize = np.prod(shapex)
        ysize = np.prod(shapey)
        #print(shapex, xsize, shapey, ysize)
        assert xsize + ysize == torch.numel(code)

        code = code.to(self.device)

        x = torch.reshape(code[:xsize], shapex)
        y = torch.reshape(code[xsize:], shapey)
        with torch.no_grad():
            out = self.model.decode([x, y])
        return torch.squeeze(out["x_hat"])

    
    available_models = {
            "models4x": models4x.SpatiallyAdaptiveCompression,
            "default": models.SpatiallyAdaptiveCompression,
        }

    @staticmethod
    def AvailableModels():
        return QmapModel.available_models.keys()

    @staticmethod
    def GetModel(model_name):
        if model_name not in QmapModel.AvailableModels():
            raise RuntimeError("Qmap model {} not found!".format(model_name))

        return QmapModel.available_models[model_name]

