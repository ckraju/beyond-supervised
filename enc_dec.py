import torch.nn as nn
import math
class EncoderDecoder(nn.Module):
    
    def __init__(self, num_classes=3):
        super(EncoderDecoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.e_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.e_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn2 = nn.BatchNorm2d(32)
        
        self.e_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn3 = nn.BatchNorm2d(64)
        
        self.e_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn4 = nn.BatchNorm2d(64)
        
        ##################################################################################
        
        self.d_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.d_bn1 = nn.BatchNorm2d(64)
        
        self.d_conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.d_bn2 = nn.BatchNorm2d(32)
        
        self.d_conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.d_bn3 = nn.BatchNorm2d(16)
        
        self.d_conv4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=0, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def encode(self, x):
        x = self.relu(self.e_bn1(self.e_conv1(x)))
        x = self.relu(self.e_bn2(self.e_conv2(x)))
        x = self.relu(self.e_bn3(self.e_conv3(x)))
        x = self.relu(self.e_bn4(self.e_conv4(x)))
        
        return x
    
    def decode(self, x):
        x = self.relu(self.d_bn1(self.d_conv1(self.upsample(x))))
        x = self.relu(self.d_bn2(self.d_conv2(self.upsample(x))))
        x = self.relu(self.d_bn3(self.d_conv3(self.upsample(x))))
        x = self.d_conv4(self.upsample(x))
        
        return x
    
    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        
        return dec
    
def encoder_decoder(pretrained=False, **kwargs):
    
    model = EncoderDecoder(**kwargs)
    return model
        
        
class Encoder(nn.Module):
    
    def __init__(self, num_classes=3):
        super(Encoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.e_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.e_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn2 = nn.BatchNorm2d(32)
        
        self.e_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn3 = nn.BatchNorm2d(64)
        
        self.e_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.e_bn4 = nn.BatchNorm2d(64)
        
        self.decoder = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def encode(self, x):
        x = self.relu(self.e_bn1(self.e_conv1(x)))
        x = self.relu(self.e_bn2(self.e_conv2(x)))
        x = self.relu(self.e_bn3(self.e_conv3(x)))
        x = self.relu(self.e_bn4(self.e_conv4(x)))
        
        return x
    
    def forward(self, x):
        enc = self.encode(x)
        if self.decoder is not None:
            enc = self.decoder(enc)
        return enc
    
def encoder(pretrained=False, **kwargs):
    model = Encoder(**kwargs)
    return model
        
        
        