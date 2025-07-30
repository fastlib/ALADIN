import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import os

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_pool_op, get_matching_convtransp, get_matching_dropout, maybe_convert_scalar_to_list, get_default_network_config
from dynamic_network_architectures.architectures.resnet import ResNet34
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU, StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from net1d import Net1D


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        # x shape: [batch_size, time_steps, input_features]
        x = x.permute(0, 2, 1)
        batch_size, time_steps, input_features = x.size()

        # Reshape to merge batch and time dimensions
        x = x.reshape(-1, input_features)  # Shape: [batch_size * time_steps, input_features]

        # Apply the layer
        x = self.layer(x)  # Shape: [batch_size * time_steps, output_features]

        # Reshape back to [batch_size, time_steps, output_features]
        output_features = x.size(-1)
        x = x.reshape(batch_size, time_steps, output_features)
        return x

class HannunNet(nn.Module):
    def __init__(self, n_classes: int, n_input_channel: int = 3, stochastic_depth_p=0.0, squeeze_excitation=False,
                 squeeze_excitation_rd_ratio=1./16):
        """
        Implements ResNetD (https://arxiv.org/pdf/1812.01187.pdf).
        Args:
            n_classes: Number of classes
            n_input_channel: Number of input channels (e.g. 3 for RGB)
            config: Configuration of the ResNet
            input_dimension: Number of dimensions of the data (1, 2 or 3)
            final_layer_dropout: Probability of dropout before the final classifier
            stochastic_depth_p: Stochastic Depth probability
            squeeze_excitation: Whether Squeeze and Excitation should be applied
            squeeze_excitation_rd_ratio: Squeeze and Excitation Reduction Ratio
        Returns:
            ResNet Model
        """
        super().__init__()
        self.name = "HannunNet"
        self.input_channels = n_input_channel
        self.cfg = {
            'features_per_stage': (32, 32, 64, 64, 128, 128, 256, 256), 
            'n_blocks_per_stage': (2, 2, 2, 2, 2, 2, 2, 2), 
            'strides': (2, 2, 2, 2, 2, 2, 2, 2),
            'dropout_p': 0.2,
            'block': BasicBlockD, 
            'bottleneck_channels': None, 
            'disable_default_stem': False, 
            'stem_channels': None
        }
        self.ops = get_default_network_config(dimension=1)

        encoder_input_features = n_input_channel
        self.stem = None

        self.encoder = ResidualEncoder(encoder_input_features, n_stages=len(self.cfg['features_per_stage']),
                                       features_per_stage=self.cfg['features_per_stage'], conv_op=self.ops['conv_op'],
                                       kernel_sizes=17, strides=self.cfg['strides'],
                                       n_blocks_per_stage=self.cfg['n_blocks_per_stage'], conv_bias=False,
                                       norm_op=self.ops['norm_op'], norm_op_kwargs=None, dropout_op=self.ops['dropout_op'],
                                       dropout_op_kwargs={'p': self.cfg["dropout_p"]}, nonlin=nn.ReLU,
                                       nonlin_kwargs={'inplace': True}, block=self.cfg['block'],
                                       bottleneck_channels=self.cfg['bottleneck_channels'], return_skips=False,
                                       disable_default_stem=self.cfg['disable_default_stem'],
                                       stem_channels=self.cfg['stem_channels'],
                                       stochastic_depth_p=0.0,
                                       squeeze_excitation=squeeze_excitation,
                                       squeeze_excitation_reduction_ratio=squeeze_excitation_rd_ratio)

        self.gap = get_matching_pool_op(conv_op=self.ops['conv_op'], adaptive=True, pool_type='avg')(1)
        self.classifier = TimeDistributed(nn.Linear(self.cfg['features_per_stage'][-1], n_classes, True))

    def forward(self, x):
        x = self.encoder(x)
        #x = self.gap(x)
        #print(x.shape)
        x = self.classifier(x)

        return x

    def embed(self, x):
        x = self.encoder(x)
        return x

class ECGFounderNet(nn.Module):
    def __init__(self, n_classes: int, n_input_channel: int = 3, features=False):
        
        super().__init__()

        self.name = "ECGFounderNet"
        self.return_features = features

        self.model = Net1D(
            in_channels=n_input_channel, 
            base_filters=64, #32 64
            ratio=1, 
            filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
            m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
            kernel_size=16, 
            stride=2, 
            groups_width=16,
            verbose=False, 
            use_bn=False,
            use_do=False,
            return_features=features,
            n_classes=150)

        if os.path.exists("/home/lukas/UU/ASRA/ALADINv2/benchmark/weights/1_lead_ECGFounder.pth"):
            print("Loading pretrained weights")
            checkpoint = torch.load("/home/lukas/UU/ASRA/ALADINv2/benchmark/weights/1_lead_ECGFounder.pth")
            state_dict = checkpoint['state_dict']

            #state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("WARNING: No pretrained weights found. This is not a problem if you use the finetuned checkpoint. If you want to finetune yourself, this is a problem! In that case, download the pretrained model weights from figshare.")

        #self.model.dense = nn.Linear(self.model.dense.in_features, n_classes)

        # for name, param in self.model.named_parameters():
        #     if 'dense' not in name:  # no freezing last layer
        #         param.requires_grad = True

    def forward(self, x):
        if self.return_features:
            x, features = self.model(x)
            return x, features
        
        return self.model(x)
    
    def embed(self, x):
        if self.return_features:
            x, features = self.model(x)
            return features
        
        return None

class ALADINNet(nn.Module):
    def __init__(self, n_classes: int, n_input_channel: int = 3, stochastic_depth_p=0.0, squeeze_excitation=False,
                 squeeze_excitation_rd_ratio=1./16):
        super().__init__()
        self.input_channels = n_input_channel

        self.cfg = {
            'features_per_stage': (32, 64, 128, 256), 
            'n_blocks_per_stage': (2, 2, 2, 2), 
            'strides': (4, 4, 4, 4),
            'dropout_p': 0.2,
            'block': BasicBlockD, 
            'bottleneck_channels': None, 
            'disable_default_stem': False, 
            'stem_channels': None
        }

        self.ops = get_default_network_config(dimension=1)

        encoder_input_features = n_input_channel
        encoder_output_channels = 6
        self.stem = None

        plan_configuration = load_json(os.path.join(nnUNet_results, "Dataset301_all_0", "ClassificationTrainer__nnUNetWithClassificationPlans__1d", "plans.json"))
        plan_configuration = plan_configuration['configurations']["1d"]
        architecture = plan_configuration['architecture']


        self.preencoder = get_network_from_plans(architecture["network_class_name"],
                architecture["arch_kwargs"],
                architecture["_kw_requires_import"],
                encoder_input_features,
                encoder_output_channels)

        print(self.preencoder.encoder)

        self.encoder = ResidualEncoder(6, n_stages=len(self.cfg['features_per_stage']),
                                       features_per_stage=self.cfg['features_per_stage'], conv_op=self.ops['conv_op'],
                                       kernel_sizes=5, strides=self.cfg['strides'],
                                       n_blocks_per_stage=self.cfg['n_blocks_per_stage'], conv_bias=False,
                                       norm_op=self.ops['norm_op'], norm_op_kwargs=None, dropout_op=self.ops['dropout_op'],
                                       dropout_op_kwargs={'p': self.cfg["dropout_p"]}, nonlin=nn.ReLU,
                                       nonlin_kwargs={'inplace': True}, block=self.cfg['block'],
                                       bottleneck_channels=self.cfg['bottleneck_channels'], return_skips=False,
                                       disable_default_stem=self.cfg['disable_default_stem'],
                                       stem_channels=self.cfg['stem_channels'],
                                       stochastic_depth_p=0.0,
                                       squeeze_excitation=squeeze_excitation,
                                       squeeze_excitation_reduction_ratio=squeeze_excitation_rd_ratio)

        self.gap = get_matching_pool_op(conv_op=self.ops['conv_op'], adaptive=True, pool_type='avg')(1)
        self.classifier = TimeDistributed(nn.Linear(self.cfg['features_per_stage'][-1], n_classes, True))

    def forward(self, x):
        #print(x.shape)
        x, _ = self.preencoder(x)
        #print(x.shape)

        #x = self.preencoder.concatenate(skips)
        #x = self.gap(x).permute(0, 2, 1)
        #print(x.shape)
        x = self.encoder(x.detach())
        #print(x.shape)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)


        return x

    def embed(self, x):
        skips = self.preencoder.encoder(x)
        x = self.preencoder.decoder.embed(skips)[-1]
        #print(x.shape)
        #x = self.gap(x).permute(0, 2, 1)
        x = self.encoder(x.detach())
        print(x.shape)
        return x

class ECGAutoencoder(nn.Module):
    def __init__(self, input_size=1792, latent_dim=32):
        super(ECGAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 896),
            nn.ReLU(),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Linear(448, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 448),
            nn.ReLU(),
            nn.Linear(448, 896),
            nn.ReLU(),
            nn.Linear(896, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

if __name__ == "__main__":
    testinp = torch.rand((1, 1, 2048))
    mdl = ALADINNet(12, 1)
    mdl(testinp)

    # mdl2 = HannunNet(11, 1)
    # mdl2(testinp)
