import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """
    Attention mechanism
    """
    def __init__(self, d_in_channels, e_in__channels, out_channels, kernel_size=(1,1), stride=(1,1)):
        super().__init__()
        if out_channels == 0:
            out_channels = 1
            
        self.We = nn.Conv2d(
                in_channels  = e_in__channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride = stride)
        
        self.Wd = nn.Conv2d(
                in_channels  = d_in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride = stride)

        self.sigmoid = nn.Sigmoid()

        self.Watt = nn.Conv2d(
                in_channels  = out_channels,
                out_channels = 1,
                kernel_size  = kernel_size,
                stride = stride)

    def forward(self, d,e):

        WeE = self.We(e)
        WdD = self.Wd(d)

        B = self.sigmoid(WeE + WdD)    # B = sigmoid(We*e+Wd*d)
        A = self.sigmoid(self.Watt(B)) # A = sigmoid(Watt*B)

        new_skip_e = torch.mul(A, e)

        return new_skip_e


class CausalConvBlock(nn.Module):
    """
    Each block in the encoder consists of a Conv2d layer followed by batch normalization, 
    dropout, and a `LeakyRelu' activation function. In addition, an attention block within the skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class CausalTransConvBlock(nn.Module):
    """
    Each block in the decoder consists of a ConvTranspose2d layer followed by batch normalization, 
    dropout, and a `LeakyRelu' activation function. In addition, an attention block within the skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

        self.att_block = AttentionBlock(in_channels, out_channels, out_channels//2)

    def forward(self, x, skip, EnableSkipAtt):
        if EnableSkipAtt:
            skip = self.att_block(x,skip)
        
        return self.conv(torch.cat((x, skip), 1)),skip


class UNET(nn.Module):
    """
    UNET architecture

    Args:
        in_channel (int): Number of input channels.
        activation (int): Activation fnction at the end of the UNET.
        EnableSkipAttention (True/False): Flag to enable attention in the skip connections.
    """
    def __init__(self, in_channel, activation = 'sigmoid', EnableSkipAttention = 0):
        super(UNET, self).__init__()
        self.EnableSkipAttention = EnableSkipAttention

        # Encoder blocks
        self.conv_block_1 = CausalConvBlock(in_channel, 32, (6, 3), (2, 2))
        self.conv_block_2 = CausalConvBlock(32,  32,  (7, 4), (2, 2))
        self.conv_block_3 = CausalConvBlock(32,  64,  (7, 5), (2, 2))
        self.conv_block_4 = CausalConvBlock(64,  64,  (6, 6), (2, 2))
        self.conv_block_5 = CausalConvBlock(64,  96,  (6, 6), (2, 2))
        self.conv_block_6 = CausalConvBlock(96,  96,  (6, 6), (2, 2))
        self.conv_block_7 = CausalConvBlock(96,  128, (2, 2), (2, 2))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1))

        # Decoder blocks
        self.tran_conv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2,2),
                stride=(1,1)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True))
        self.tran_conv_block_2 = CausalTransConvBlock(256, 128, (2, 2), (2, 2))
        self.tran_conv_block_3 = CausalTransConvBlock(128, 96,  (6, 6), (2, 2))
        self.tran_conv_block_4 = CausalTransConvBlock(96,  96,  (6, 6), (2, 2))
        self.tran_conv_block_5 = CausalTransConvBlock(96,  64,  (6, 6), (2, 2))
        self.tran_conv_block_6 = CausalTransConvBlock(64,  64,  (7, 5), (2, 2))
        self.tran_conv_block_7 = CausalTransConvBlock(64,  32,  (7, 4), (2, 2))
        self.tran_conv_block_8 = CausalTransConvBlock(32,  32,  (6, 3), (2, 2))

        if EnableSkipAttention == 0: # 'Standard' mode
            self.last_conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=in_channel,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm2d(num_features=in_channel),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(inplace=True))
        else:
            self.last_conv_block = CausalTransConvBlock(32, in_channel, (1, 1), (1, 1))

        # Dense layer    
        if activation == 'tanh':
            self.dense = nn.Sequential(nn.Linear(514, 514),nn.Tanh())
        else:
            self.dense = nn.Sequential(nn.Linear(514, 514),nn.Sigmoid())

    def forward(self, x):
        skip = torch.zeros_like(x)
        
        # Encoder blocks
        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        if self.EnableSkipAttention == 0: # Standard mode
            EnableSkipAtt = 0
        else: # Attention mode
            EnableSkipAtt = 1

        # Decoder blocks
        d = self.tran_conv_block_1(e8)
        d,_ = self.tran_conv_block_2(d, e7, EnableSkipAtt)
        d,_ = self.tran_conv_block_3(d, e6, EnableSkipAtt)
        d,_ = self.tran_conv_block_4(d, e5, EnableSkipAtt)
        d,_ = self.tran_conv_block_5(d, e4, EnableSkipAtt)
        d,_ = self.tran_conv_block_6(d, e3, EnableSkipAtt)
        d,_ = self.tran_conv_block_7(d, e2, EnableSkipAtt)
        d,_ = self.tran_conv_block_8(d, e1, EnableSkipAtt)

        if self.EnableSkipAttention == 0: # Standard mode
            d = self.last_conv_block(d)
        else: # Attention mode
            d,skip = self.last_conv_block(d, x, EnableSkipAtt)

        d = d.permute(0,1,3,2)
        d = self.dense(d).permute(0,1,3,2)

        return d,skip       
    
    