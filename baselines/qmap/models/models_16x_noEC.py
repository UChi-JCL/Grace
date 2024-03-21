import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperpriors import ScaleHyperprior
from .utils import conv
from .layers import GDN1

from .layers import SFT, SFTResblk, Conv2d, UpConv2d

G_USE_SMALLER = False


class SpatiallyAdaptiveCompression(ScaleHyperprior):
    def __init__(self, N=192, M=192, sft_ks=3, prior_nc=64, **kwargs):
        super().__init__(N, M, **kwargs)
        print(" ======= INITIALIZE NETWORK WITH G_USE_SMALLER = {}".format(G_USE_SMALLER))
        ### condition networks ###
        # g_a,c
        self.qmap_feature_g1 = nn.Sequential(
            conv(4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3, stride=4),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g5_new = nn.Sequential(
            conv(prior_nc, prior_nc, 3, stride=4),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        ## h_a,c
        #self.qmap_feature_h1 = nn.Sequential(
        #    conv(M + 1, prior_nc * 4, 3, 1),
        #    nn.LeakyReLU(0.1, True),
        #    conv(prior_nc * 4, prior_nc * 2, 3, 1),
        #    nn.LeakyReLU(0.1, True),
        #    conv(prior_nc * 2, prior_nc, 3, 1)
        #)
        #self.qmap_feature_h2 = nn.Sequential(
        #    conv(prior_nc, prior_nc, 3),
        #    nn.LeakyReLU(0.1, True),
        #    conv(prior_nc, prior_nc, 1, 1)
        #)
        #self.qmap_feature_h3 = nn.Sequential(
        #    conv(prior_nc, prior_nc, 3),
        #    nn.LeakyReLU(0.1, True),
        #    conv(prior_nc, prior_nc, 1, 1)
        #)

        ## f_c
        #self.qmap_feature_gs0 = nn.Sequential(
        #    UpConv2d(N, N//2, 3),
        #    nn.LeakyReLU(0.1, True),
        #    UpConv2d(N//2, N//4, stride=1),
        #    nn.LeakyReLU(0.1, True),
        #    conv(N//4, N//4, 3, 1)
        #)

        self.pre_pink_module = nn.Sequential(
            conv(1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )

        # g_s,c
        self.qmap_feature_gs1 = nn.Sequential(
            conv(M + prior_nc, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2_new = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3, stride=4),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3, stride=4),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs5 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        ### compression networks ###
        # g_a, encoder
        self.g_a = None
        self.g_a0 = Conv2d(3, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_a3 = Conv2d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv2d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a9 = Conv2d(N, N, stride=4)
        self.g_a10 = GDN1(N)
        self.g_a11 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12_new = Conv2d(N, M, stride=4)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # h_a, hyper encoder
        #self.h_a = None
        #self.h_a0 = Conv2d(M, N, kernel_size=3, stride=1)
        #self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        #self.h_a2 = nn.LeakyReLU(inplace=True)

        #self.h_a3 = Conv2d(N, N)
        #self.h_a4 = SFT(N, prior_nc, ks=sft_ks)
        #self.h_a5 = nn.LeakyReLU(inplace=True)

        #self.h_a6 = Conv2d(N, N)
        #self.h_a7 = SFTResblk(N, prior_nc, ks=sft_ks)
        #self.h_a8 = SFTResblk(N, prior_nc, ks=sft_ks)

        # g_s, decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2_new = UpConv2d(M, N, stride=4)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv2d(N, N, stride=4)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv2d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s11 = UpConv2d(N // 2, N // 4)
        self.g_s12 = GDN1(N // 4, inverse=True)
        self.g_s13 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_s14 = Conv2d(N // 4, 3, kernel_size=5, stride=1)

        ## h_s, hyper decoder
        #self.h_s = nn.Sequential(
        #    UpConv2d(N, M),
        #    nn.LeakyReLU(inplace=True),
        #    UpConv2d(M, M * 3 // 2, stride=1),
        #    nn.LeakyReLU(inplace=True),
        #    conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        #)
        

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(torch.cat([qmap, x], dim=1))
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a9(x)
        x = self.g_a10(x)
        x = self.g_a11(x, qmap)

        qmap = self.qmap_feature_g5_new(qmap)
        x = self.g_a12_new(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    #def h_a(self, x, qmap):
        #qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        #qmap = self.qmap_feature_h1(torch.cat([qmap, x], dim=1))
        #x = self.h_a0(x)
        #x = self.h_a1(x, qmap)
        #x = self.h_a2(x)

        ##qmap = self.qmap_feature_h2(qmap)
        ##x = self.h_a3(x)
        ##x = self.h_a4(x, qmap)
        ##x = self.h_a5(x)

        #qmap = self.qmap_feature_h3(qmap)
        #x = self.h_a6(x)
        #x = self.h_a7(x, qmap)
        #x = self.h_a8(x, qmap)
        #return x

    def g_s(self, x, z):
        w = F.adaptive_avg_pool2d(z, x.size()[2:])
        w = self.pre_pink_module(w)
        w = self.qmap_feature_gs1(torch.cat([w, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        w = self.qmap_feature_gs2_new(w)
        x = self.g_s2_new(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        w = self.qmap_feature_gs4(w)
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        w = self.qmap_feature_gs5(w)
        x = self.g_s11(x)
        x = self.g_s12(x)
        x = self.g_s13(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        #import pdb
        #pdb.set_trace()
        y = self.g_a(x, qmap)
        #z = self.h_a(y, qmap)
        #z_hat, z_likelihoods = self.entropy_bottleneck(z)

        #gaussian_params = self.h_s(z_hat)
        #scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        yq = self.entropy_bottleneck.quantize(y, "symbols", None)
        x_hat = self.g_s(yq, qmap)

        return {
            "x_hat": x_hat,
            #"likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        raise RuntimeError("ERROR: do not have compression module in this model!")
        return None

        #y = self.g_a(x, qmap)
        #z = self.h_a(y, qmap)

        #z_strings = self.entropy_bottleneck.compress(z)
        #z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        #gaussian_params = self.h_s(z_hat)
        #scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #indexes = self.gaussian_conditional.build_indexes(scales_hat)
        #import pdb
        #pdb.set_trace()
        #y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        #return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        raise RuntimeError("ERROR: do not have compression module in this model!")
        return None

        #assert isinstance(strings, list) and len(strings) == 2
        #z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        #gaussian_params = self.h_s(z_hat)
        #scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #indexes = self.gaussian_conditional.build_indexes(scales_hat)
        #y_hat = self.gaussian_conditional.decompress(
        #    strings[0], indexes, means=means_hat
        #)
        #x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        #return {"x_hat": x_hat}

    def encode(self, x, qmap):
        y = self.g_a(x, qmap)
        
        yq = self.entropy_bottleneck.quantize(y, "symbols", None)
        return {"strings": [yq, zq], "shape": z.size()[-2:]}

    def decode(self, strings, qmap = None):
        if qmap == None:
            raise RuntimeError("Need qmap to decode in this model!")
        y_hat = strings[0]

        y_hatq = self.entropy_bottleneck.dequantize(y_hat, None)

        x_hat = self.g_s(y_hatq, qmap).clamp_(0, 1)
        return {"x_hat": x_hat}

    def load_from_state_dict(self, state_dict):
        own_state = self.state_dict()
        #print(state_dict.keys())
        load = 0
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if param.size() != own_state[name].size():
                continue
            if isinstance(param, nn.Parameter):
                param = param.data

            #print(name, param.shape, own_state[name].shape)
            own_state[name].copy_(param)
            load += 1

        print(" ==== loading {} from older state_dict ({}), {} parameters in the current model\n".format(load, len(state_dict), len(own_state)))

