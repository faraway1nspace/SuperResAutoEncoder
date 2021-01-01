import torch

class HyperParams:
    def __init__(self, params_dict):
        for k,v in params_dict.items():
            self.__setattr__(k,v)

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class TransformerNet2(torch.nn.Module):
    """ includes two skip connections"""
    def __init__(self, params):
        super(TransformerNet2, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32,
                               kernel_size = params.conv1_kernel_size1,
                               stride=1,
                               extra_padding = params.conv1_extra_padding)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        #self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        #self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(params.resblock_dim)
        self.res2 = ResidualBlock(params.resblock_dim)
        self.res3 = ResidualBlock(params.resblock_dim)
        #self.res4 = ResidualBlock(params.resblock_dim)
        #self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(params.deconv1_inchannel_size, params.deconv1_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(params.deconv1_outchannel_size, affine=True)
        self.deconv2 = UpsampleConvLayer(params.deconv2_inchannel_size+params.input_inchannel_size, params.deconv2_outchannel_size, kernel_size=3, stride=1, upsample=4)
        self.in5 = torch.nn.InstanceNorm2d(params.deconv2_outchannel_size, affine=True)
        self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size+params.input_inchannel_size, params.deconv3_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in6 = torch.nn.InstanceNorm2d(params.deconv3_outchannel_size, affine=True)
        self.deconv_out = ConvLayer(params.deconv3_outchannel_size, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # final downsampling for the content loss        
        self.downsample_mode = params.downsample_mode
        self.downsample_factor = params.downsample_factor
    
    def forward(self, X):
        # straight upsampling of the input by x4
        X_upsampled_x4 = torch.nn.functional.interpolate(X, mode='nearest', scale_factor=4)
        # convolution
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        #y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        #y = self.res4(y)
        #y = self.res5(y)
        y_back_to_orig_dim = self.relu(self.in4(self.deconv1(y))) # in[32,32] out [64,64]
        y_cat = torch.cat([X, y_back_to_orig_dim],axis=1)
        y_dim_x4 = self.relu(self.in5(self.deconv2(y_cat))) # in [64,64] out [256,256]
        y_cat = torch.cat([X_upsampled_x4, y_dim_x4], axis=1)
        y_dim_x8 = self.relu(self.in6(self.deconv3(y_cat))) # in [256,256] out[512,512]
        out = self.deconv_out(y_dim_x8)
        out_downsampled = torch.nn.functional.interpolate(out, mode=self.downsample_mode, scale_factor=self.downsample_factor)
        return out, out_downsampled

class TransformerNet3(torch.nn.Module):
    """ includes just one skip connection"""
    def __init__(self, params):
        super(TransformerNet3, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32,
                               kernel_size = params.conv1_kernel_size1,
                               stride=1,
                               extra_padding = params.conv1_extra_padding)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        #self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        #self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(params.resblock_dim)
        self.res2 = ResidualBlock(params.resblock_dim)
        self.res3 = ResidualBlock(params.resblock_dim)
        #self.res4 = ResidualBlock(params.resblock_dim)
        #self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(params.deconv1_inchannel_size, params.deconv1_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(params.deconv1_outchannel_size, affine=True)
        self.deconv2 = UpsampleConvLayer(params.deconv2_inchannel_size+params.input_inchannel_size, params.deconv2_outchannel_size, kernel_size=3, stride=1, upsample=4)
        self.in5 = torch.nn.InstanceNorm2d(params.deconv2_outchannel_size, affine=True)
        #self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size+params.input_inchannel_size, params.deconv3_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size, params.deconv3_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in6 = torch.nn.InstanceNorm2d(params.deconv3_outchannel_size, affine=True)
        self.deconv_out = ConvLayer(params.deconv3_outchannel_size, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # final downsampling for the content loss        
        self.downsample_mode = params.downsample_mode
        self.downsample_factor = params.downsample_factor
    
    def forward(self, X):
        # straight upsampling of the input by x4
        #X_upsampled_x4 = torch.nn.functional.interpolate(X, mode='nearest', scale_factor=4)
        # convolution
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        #y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        #y = self.res4(y)
        #y = self.res5(y)
        y_back_to_orig_dim = self.relu(self.in4(self.deconv1(y))) # in[32,32] out [64,64]
        y_cat = torch.cat([X, y_back_to_orig_dim],axis=1)
        y_dim_x4 = self.relu(self.in5(self.deconv2(y_cat))) # in [64,64] out [256,256]
        #y_cat = torch.cat([X_upsampled_x4, y_dim_x4], axis=1)
        y_dim_x8 = self.relu(self.in6(self.deconv3(y_dim_x4))) # in [256,256] out[512,512]
        out = self.deconv_out(y_dim_x8)
        out_downsampled = torch.nn.functional.interpolate(out, mode=self.downsample_mode, scale_factor=self.downsample_factor)
        return out, out_downsampled

class TransformerNet4(torch.nn.Module):
    """ includes just one skip connection"""
    def __init__(self, params):
        super(TransformerNet4, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32,
                               kernel_size = params.conv1_kernel_size1,
                               stride=1,
                               extra_padding = params.conv1_extra_padding)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        #self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        #self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(params.resblock_dim)
        self.res2 = ResidualBlock(params.resblock_dim)
        self.res3 = ResidualBlock(params.resblock_dim)
        #self.res4 = ResidualBlock(params.resblock_dim)
        #self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(params.deconv1_inchannel_size, params.deconv1_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(params.deconv1_outchannel_size, affine=True)
        self.deconv2 = UpsampleConvLayer(params.deconv2_inchannel_size+params.input_inchannel_size, params.deconv2_outchannel_size, kernel_size=3, stride=1, upsample=4)
        self.in5 = torch.nn.InstanceNorm2d(params.deconv2_outchannel_size, affine=True)
        #self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size+params.input_inchannel_size, params.deconv3_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size, params.deconv3_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in6 = torch.nn.InstanceNorm2d(params.deconv3_outchannel_size, affine=True)
        self.deconv_out = ConvLayer(params.deconv3_outchannel_size, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # final downsampling for the content loss        
        #self.downsample_mode = params.downsample_mode
        #self.downsample_factor = params.downsample_factor
    
    def forward(self, X):
        # straight upsampling of the input by x4
        #X_upsampled_x4 = torch.nn.functional.interpolate(X, mode='nearest', scale_factor=4)
        # convolution
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        #y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        #y = self.res4(y)
        #y = self.res5(y)
        y_back_to_orig_dim = self.relu(self.in4(self.deconv1(y))) # in[32,32] out [64,64]
        y_cat = torch.cat([X, y_back_to_orig_dim],axis=1)
        y_dim_x4 = self.relu(self.in5(self.deconv2(y_cat))) # in [64,64] out [256,256]
        #y_cat = torch.cat([X_upsampled_x4, y_dim_x4], axis=1)
        y_dim_x8 = self.relu(self.in6(self.deconv3(y_dim_x4))) # in [256,256] out[512,512]
        out = self.deconv_out(y_dim_x8)
        #out_downsampled = torch.nn.functional.interpolate(out, mode=self.downsample_mode, scale_factor=self.downsample_factor)
        return out, None

    
class NetResidUpsample(torch.nn.Module):
    """ blends upsampling and res blocks to add features and depth at same time"""
    def __init__(self, params):
        super(NetResidUpsample, self).__init__()
        
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size = params.conv1_kernel_size1, stride=2, extra_padding = params.conv1_extra_padding)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True) # out 32 x 32
        self.conv2 = ConvLayer(32, 64, kernel_size=params.conv2_kernel_size1, stride=2) # out 16 x 16
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, params.conv3_outchannels-3, kernel_size=3, stride=2) # out 8x8
        self.in3 = torch.nn.InstanceNorm2d(params.conv3_outchannels-3, affine=True)
        
        # zeroth upsample back to 64x64 -> concatenate with original
        self.res1 = ResidualBlock(params.conv3_outchannels)
        
        # first upsample : up to 128 (notice -3 because of concatenation
        self.deconv1 = UpsampleConvLayer(params.deconv1_inchannel_size, params.deconv1_outchannel_size-3, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(params.deconv1_outchannel_size-3, affine=True)
        self.res2 = ResidualBlock(params.deconv1_outchannel_size)
        
        # 2nd upsample : up to 256
        self.deconv2 = UpsampleConvLayer(params.deconv2_inchannel_size, params.deconv2_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(params.deconv2_outchannel_size, affine=True)
        self.res3 = ResidualBlock(params.deconv2_outchannel_size) # receives upsampled X
        
        # 2nd upsample : up to 512
        self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size, params.deconv3_outchannel_size, kernel_size=params.deconv3_kernel_size, stride=1, upsample=2)
        self.in6 = torch.nn.InstanceNorm2d(params.deconv3_outchannel_size, affine=True)
        self.res4 = ResidualBlock(params.deconv3_outchannel_size)
        
        # final out
        self.deconv_out = ConvLayer(params.deconv3_outchannel_size, 3, kernel_size=9, stride=1)
        
        # Non-linearities
        self.relu = torch.nn.ReLU()
    
    def forward(self, X):
        
        # down-convolution 1 # out 32x32
        y = self.relu(self.in1(self.conv1(X)))
        
        # down-convolution 2 # out 16x16
        y = self.relu(self.in2(self.conv2(y)))
        
        # down-convolution 3 # out 8x8
        y_8 = self.relu(self.in3(self.conv3(y)))
        
        # zeroth upsample back to original
        y_64 = torch.nn.functional.interpolate(y_8, mode='nearest', scale_factor=8) # back up to 64
        # concatenate convolutions and original
        y_cat = torch.cat([y_64, X], axis=1) #
        y_64 = self.res1(y_cat)
        
        # upsample: upto 128
        y_128 = self.relu(self.in4(self.deconv1(y_64))) # 64->182
        X_128 = torch.nn.functional.interpolate(X, mode='nearest', scale_factor=2) # scale original upto 128
        y_cat128 = torch.cat([y_128, X_128], axis=1)
        y = self.res2(y_cat128)
        
        # upsample: upto 256
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.res3(y)
        
        # upsample: upto 512
        y = self.relu(self.in6(self.deconv3(y)))
        y = self.res4(y)
        
        # final convolution
        out = self.deconv_out(y)
        return out, None

class ResidUpscale256(torch.nn.Module):
    """ blends upsampling and res blocks to add features and depth at same time"""
    def __init__(self, params):
        super(ResidUpscale256, self).__init__()
        # parameters
        self.downsample_factor = params.downsample_factor
        self.downsample_mode = params.downsample_mode
        
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size = params.conv1_kernel_size1, stride=2, extra_padding = params.conv1_extra_padding)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True) # out 32 x 32
        self.conv2 = ConvLayer(32, 64, kernel_size=params.conv2_kernel_size1, stride=2) # out 16 x 16
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, params.conv3_outchannels, kernel_size=3, stride=2) # out 8x8
        self.in3 = torch.nn.InstanceNorm2d(params.conv3_outchannels, affine=True)
        
        # residual block at size 8x8
        self.res1 = ResidualBlock(params.conv3_outchannels)
        
        # first upsample : from 8x8 to 16x16
        self.deconv1 = UpsampleConvLayer(params.deconv1_inchannel_size, params.deconv1_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(params.deconv1_outchannel_size, affine=True)
        self.res2 = ResidualBlock(params.deconv1_outchannel_size)
        
        # 2nd upsample : up to 32x32
        self.deconv2 = UpsampleConvLayer(params.deconv2_inchannel_size, params.deconv2_outchannel_size, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(params.deconv2_outchannel_size, affine=True)
        self.res3 = ResidualBlock(params.deconv2_outchannel_size) # receives upsampled X
        
        # 3rd upsample : up to 64x64
        self.deconv3 = UpsampleConvLayer(params.deconv3_inchannel_size, params.deconv3_outchannel_size-3, kernel_size=params.deconv3_kernel_size, stride=1, upsample=2)
        self.in6 = torch.nn.InstanceNorm2d(params.deconv3_outchannel_size-3, affine=True)
        self.res4 = ResidualBlock(params.deconv3_outchannel_size)
        
        # 4th upsample : up to 128x128
        self.deconv4 = UpsampleConvLayer(params.deconv4_inchannel_size, params.deconv4_outchannel_size, kernel_size=params.deconv4_kernel_size, stride=1, upsample=2)
        self.in7 = torch.nn.InstanceNorm2d(params.deconv4_outchannel_size, affine=True)
        self.res5 = ResidualBlock(params.deconv4_outchannel_size)
        
        # 5th upsample : up to 256x256
        self.deconv5 = UpsampleConvLayer(params.deconv5_inchannel_size, params.deconv5_outchannel_size, kernel_size=params.deconv5_kernel_size, stride=1, upsample=2)
        self.in8 = torch.nn.InstanceNorm2d(params.deconv5_outchannel_size, affine=True)
        self.res6 = ResidualBlock(params.deconv5_outchannel_size)
        
        # final out
        self.deconv_out = ConvLayer(params.deconv5_outchannel_size, 3, kernel_size=9, stride=1)
        
        # Non-linearities
        self.relu = torch.nn.ReLU()
    
    def forward(self, X):
        
        # down-convolution 1 # out 32x32
        y = self.relu(self.in1(self.conv1(X)))
        
        # down-convolution 2 # out 16x16
        y = self.relu(self.in2(self.conv2(y)))
        
        # down-convolution 3 # out 8x8
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        
        # upscale 1
        y = self.relu(self.in4(self.deconv1(y))) # 8x8->16x16
        y = self.res2(y)
        
        # upscale 2
        y = self.relu(self.in5(self.deconv2(y))) # 16x16->32x32
        y = self.res3(y)
        
        # upscale 3 (back to input-size) (concatenate with input)
        y = self.relu(self.in6(self.deconv3(y))) # 32x32->64x64
        y_cat = torch.cat([y, X], axis=1) #
        y = self.res4(y_cat)
        
        # upscale 4
        y = self.relu(self.in7(self.deconv4(y))) # 64x64->128x128
        y = self.res5(y)
        
        # upscale 5
        y = self.relu(self.in8(self.deconv5(y))) # 128x128->256x256
        y = self.res6(y)
                
        # final convolution
        out = self.deconv_out(y)
        
        # downsample output (for downscaled loss)
        out_downsampled = torch.nn.functional.interpolate(out, mode=self.downsample_mode, scale_factor=self.downsample_factor)
        
        return out, out_downsampled

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, extra_padding=None):
        super(ConvLayer, self).__init__()
        if extra_padding is None:
            extra_padding = 0
        reflection_padding = kernel_size // 2 + extra_padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels, kernel_size = None):
        super(ResidualBlock, self).__init__()
        if kernel_size is None:
            kernel_size = 3
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
