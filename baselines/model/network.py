''' 
Spatial-Temporal Transformer Networks
https://github.com/researchmm/STTN

For our error compensation network, we slightly change input and output settings from STTN 
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from error_concealment.spectral_norm import spectral_norm as _spectral_norm
from .ops import *

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class ErrorCompensationNetwork(BaseNetwork):
    def __init__(self, args, init_weights=True):
        super(ErrorCompensationNetwork, self).__init__()
        self.args = args
        stack_num = self.args.stack_num
        channel = self.args.channels*self.args.patch_size_num
        err_channel = 3
    
        in_channels = 3 + err_channel + 1 # RGB, Error, error_mask
    
        # error completion network 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        patch_size = [] 
        for f in range(self.args.patch_size_num):
            patch_size.append(1/(math.pow(2,f+1)))

        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(self.args, patch_size, hidden=channel, in_channels=3))
        self.transformer = nn.Sequential(*blocks)


        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            GatedDeConv2d(channel, 128, kernel_size=3, padding=1, activation='lrelu'),
            GatedConv2d(128, 64, kernel_size=3, stride=1, padding=1, activation='lrelu'),
            GatedDeConv2d(64, 64, kernel_size=3, padding=1, activation='lrelu'),
            GatedConv2d(64, 3, kernel_size=3, stride=1, padding=1, activation='none')
        )

        if init_weights:
            self.init_weights()

    def forward(self, ofilled_frames, err_guidances, rem_masks, err_masks, prop_masks, empty_err_masks):
        # extracting features

        
        b, t, c, h, w = ofilled_frames.size()

        ofilled_frames = ofilled_frames.view(b*t, 3, h, w)
        err_guidances = err_guidances.view(b*t, 3, h, w)
        rem_masks = rem_masks.view(b*t, 1, h, w)
        err_masks = err_masks.view(b*t, 1, h, w)
        empty_err_masks = empty_err_masks.view(b*t, 1, h, w)
        prop_masks = prop_masks.view(b*t, 1, h, w)


        
        # error completion 
        err_info = torch.cat((err_guidances, err_masks), 1)
          
        input = torch.cat((ofilled_frames, err_info), 1)
    
        enc_feat = self.encoder(input)
        _,c,_,_ = enc_feat.size()
              

        mask = torch.cat((rem_masks, prop_masks, empty_err_masks), 1)
        mask = F.interpolate(mask, scale_factor=1.0/4)
        

        enc_feat = self.transformer(
            {'x': enc_feat, 'm':  mask, 'b': b, 'c': c})['x']
            
        dec_feat = self.decoder(enc_feat)
        err_outs = 2*torch.tanh(dec_feat) # [-2 to 2]

        err_outs = err_outs.view(b,t,3,h,w)

        
        return err_outs





class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################
# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            GatedConv2d(d_model, d_model, kernel_size=3, stride=1, padding=2, dilation=2, activation='lrelu'),
            GatedConv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, activation='lrelu'))

    def forward(self, x):
        x = self.conv(x)
        return x

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m_query, m_key):
        
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))

        mask_scores = torch.matmul(m_query, m_key.transpose(-2, -1)
                              ) / math.sqrt(m_query.size(-1))
        scores = scores * mask_scores
        # print(scores.size(), m.size())
        # scores = scores * m 
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, args, patchsize, d_model, in_channels=1):
        super().__init__()
        self.args = args
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

      
        self.mask_query_embedding = nn.Conv2d(
            in_channels, d_model, kernel_size=1, padding=0)

        self.mask_key_embedding = nn.Conv2d(
            in_channels, d_model, kernel_size=1, padding=0)

        self.attention = Attention()

    def forward(self, x, m, b, c):

        # pad imgs
        _, _, h, w = x.size()
        

        pad_h, pad_w = math.ceil(h/64)*64 - h, math.ceil(w/64)*64 - w
        # pad_h, pad_w = math.ceil(h/32)*32 - h, math.ceil(w/32)*32 - w
        pad_up = pad_h//2
        pad_down = pad_h - pad_up
        pad_left = pad_w//2
        pad_right =  pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_up, pad_down), "constant", 0)
        m = F.pad(m, (pad_left, pad_right, pad_up, pad_down), "constant", 0)
      

        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)

        _mask_query = self.mask_query_embedding(m)
        _mask_key = self.mask_key_embedding(m)

        for scale, query, key, mask_query, mask_key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                        torch.chunk(_mask_query, len(self.patchsize), dim=1),
                                                        torch.chunk(_mask_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            width, height = int(w*scale), int(h*scale)
            out_w, out_h = w // width, h // height

            # mm = fm.view(b, 1, out_h, height, out_w, width)
            # mm = mm.permute(0, 1, 2, 4, 3, 5).contiguous().view(
            #     b,  out_h*out_w, height*width)
            # mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, out_h*out_w, 1)

            # 1) embedding and reshape
            # print(query.size(), b, d_k, out_h, height, out_w, width)
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3,5,2,4,6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)

            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3,5,2,4,6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)

            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3,5,2,4,6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)

            mask_query = mask_query.view(b, t, d_k, out_h, height, out_w, width)
            mask_query = mask_query.permute(0, 1, 3,5,2,4,6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)

            mask_key = mask_key.view(b, t, d_k, out_h, height, out_w, width)
            mask_key = mask_key.permute(0, 1, 3,5,2,4,6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value, mask_query, mask_key)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            # y = y.view(b, t, out_h, out_w, d_k, height, width)
            # y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        x = x[:,:, pad_up: h - pad_down, pad_left : w- pad_right]
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self,args, patchsize, hidden=128, in_channels=1):
        super().__init__()

            
        self.attention = MultiHeadedAttention(args, patchsize, d_model=hidden, in_channels=in_channels)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c = x['x'], x['m'], x['b'], x['c']
        x = x + self.attention(x, m, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'm':m, 'b': b, 'c': c}


###################################################


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d =spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class GatedDeConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,  pad_type = 'zero',activation=torch.nn.LeakyReLU(0.2, inplace=True), norm = 'none',sn=False, scale_factor=2):
        super(GatedDeConv2d, self).__init__()
        self.conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)
        



# ######################################################################
# ######################################################################



class Discriminator(BaseNetwork):
    def __init__(self, args, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.args = args
        nf = 64
        
        
        in_channels = 2

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        B, T, C, H, W = xs.shape
        # xs = xs.view(B*T, C,H,W)
        xs_t = torch.transpose(xs, 1, 2) # B, C, T, H, W
        # print(xs_t.size())

        out = self.conv(xs_t)
        if self.use_sigmoid:
            out = torch.sigmoid(out)

        out = torch.transpose(out, 2, 1)  # B, T, C, H, W
        # _,C,H,W = out.size()
        # out = out.view(B,T,C,H,W)

        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module

