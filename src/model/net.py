import torch
import torch.nn as nn
from torchvision.models import vgg19
from loss import adaptive_instance_normalization as adain
from loss import calc_mean_std

# Decoder definition
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, kernel_size=3),
)

# Encoder based on pre-trained VGG19
vgg = vgg19(pretrained=True).features

# Truncate the VGG layers up to relu4_1
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_1 = nn.Sequential(*vgg[:4])   # relu1_1
        self.enc_2 = nn.Sequential(*vgg[4:9]) # relu2_1
        self.enc_3 = nn.Sequential(*vgg[9:16])# relu3_1
        self.enc_4 = nn.Sequential(*vgg[16:23])# relu4_1
        # Freeze encoder weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = []
        x = self.enc_1(x); results.append(x)
        x = self.enc_2(x); results.append(x)
        x = self.enc_3(x); results.append(x)
        x = self.enc_4(x); results.append(x)
        return results

# Main Net
class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, x):
        """
        Extract relu1_1, relu2_1, relu3_1, relu4_1 from input.
        """
        return self.encoder(x)

    def encode(self, x):
        """
        Extract relu4_1 from input.
        """
        return self.encoder(x)[-1]

    def calc_content_loss(self, input, target):
        """
        Calculate content loss between input and target.
        """
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        """
        Calculate style loss between input and target.
        """
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        """
        Forward pass for style transfer.
        """
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
