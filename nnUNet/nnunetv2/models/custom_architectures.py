from typing import Union, Type, List, Tuple
import numpy as np

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_pool_op, get_matching_convtransp, maybe_convert_scalar_to_list
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.architectures.resnet import ResNet18
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU, StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Linear):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

class StackedConvBlocks3p(nn.Module):
    def __init__(self,
                    num_convs: int,
                    conv_op: Type[_ConvNd],
                    input_channels: Union[List[int], Tuple[int, ...]],
                    skip_channels: int,
                    output_channels: Union[int, List[int], Tuple[int, ...]],
                    kernel_size: Union[int, List[int], Tuple[int, ...]],
                    initial_stride: Union[int, List[int], Tuple[int, ...]],
                    conv_bias: bool = False,
                    norm_op: Union[None, Type[nn.Module]] = None,
                    norm_op_kwargs: dict = None,
                    dropout_op: Union[None, Type[_DropoutNd]] = None,
                    dropout_op_kwargs: dict = None,
                    nonlin: Union[None, Type[torch.nn.Module]] = None,
                    nonlin_kwargs: dict = None,
                    nonlin_first: bool = False
                ):

        super().__init__()

        convs = []
        for i in range(len(input_channels)):
            convs.append(conv_op(input_channels[i], skip_channels, kernel_size=kernel_size[0], padding=kernel_size[0] // 2, bias=conv_bias))

        #print("Num skip channels:", skip_channels*len(input_channels))
        #print("Num output channels:", output_channels)
        self.convs = nn.ModuleList(convs)
        self.skip_size = skip_channels
        self.decoder_size = skip_channels*len(input_channels)
        self.aggregate = ConvDropoutNormReLU(conv_op, skip_channels*len(input_channels), output_channels, kernel_size, initial_stride, conv_bias, norm_op,
                                            norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)

        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x: List[torch.Tensor]):
        
        out = []
        for i in range(len(x)):
            out.append(self.convs[i](x[i]))

        #print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape)

        out = torch.cat(out, dim=1)
        out = self.aggregate(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.aggregate.compute_conv_feature_map_size(input_size)
        for b in self.convs:
            output += input_size*self.skip_size
        return output


class UNetDecoder3p(nn.Module):
    def __init__(self,
                    encoder: Union[PlainConvEncoder, ResidualEncoder],
                    num_classes: int,
                    n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                    deep_supervision,
                    nonlin_first: bool = False,
                    norm_op: Union[None, Type[nn.Module]] = None,
                    norm_op_kwargs: dict = None,
                    dropout_op: Union[None, Type[_DropoutNd]] = None,
                    dropout_op_kwargs: dict = None,
                    nonlin: Union[None, Type[torch.nn.Module]] = None,
                    nonlin_kwargs: dict = None,
                    conv_bias: bool = None
                 ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        seg_layers = []
        n_stages_encoder = len(encoder.output_channels)
        #print(encoder.output_channels)

        self.features_decoder = encoder.output_channels[0]*n_stages_encoder

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[0]

            channels = list(encoder.output_channels[:-s]) + [self.features_decoder] * (s-1) + list(encoder.output_channels[-1:])

            stride_for_transpconv = encoder.strides[-s]
            
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks3p(
                n_conv_per_stage[s-1], 
                encoder.conv_op, 
                channels, 
                input_features_skip,
                self.features_decoder,
                encoder.kernel_sizes[-(s + 1)],
                1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(self.features_decoder, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """

        ls = [skips[-1]]
        seg_outputs = []
        for s in range(len(self.stages)):
            ind = len(self.stages) - s - 1
            #print("Ind:", ind)
            xs = []
            st = 2 ** ind
            for i in range(ind):
                x = nn.functional.max_pool1d(skips[i], kernel_size=st, stride=st)
                xs.append(x)
                st = st // 2
                #print("Shape: ",x.shape)
            xs.append(skips[ind])
            #print("Shape: ",skips[ind].shape)
            for i in range(s+1):
                x = nn.functional.interpolate(ls[-(i+1)], size=skips[ind].shape[-1], mode='linear', align_corners=False)
                xs.append(x)
                #print("Shape: ",x.shape)

            l = self.stages[s](xs)
            ls.append(l)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](l))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](l))

        seg_outputs = seg_outputs[::-1]
        #print("Seg outputs: ", seg_outputs[0].shape, seg_outputs[1].shape, seg_outputs[2].shape, seg_outputs[3].shape)

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def embed(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        ls = [skips[-1]]
        seg_outputs = []
        for s in range(len(self.stages)):
            ind = len(self.stages) - s - 1
            #print("Ind:", ind)
            xs = []
            st = 2 ** ind
            for i in range(ind):
                x = nn.functional.max_pool1d(skips[i], kernel_size=st, stride=st)
                xs.append(x)
                st = st // 2
                #print("Shape: ",x.shape)
            xs.append(skips[ind])
            #print("Shape: ",skips[ind].shape)
            for i in range(s+1):
                x = nn.functional.interpolate(ls[-(i+1)], size=skips[ind].shape[-1], mode='linear', align_corners=False)
                xs.append(x)
                #print("Shape: ",x.shape)

            l = self.stages[s](xs)
            ls.append(l)

        return ls

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class UNetDecoderExt(UNetDecoder):
    def embed(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            seg_outputs.append(x)
            lres_input = x

        return seg_outputs

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=10, num_timesteps=8, hidden_dim=256, num_heads=8, num_layers=2, conv_op=nn.Conv1d, norm_op=nn.BatchNorm1d):
        super(TransformerClassifier, self).__init__()

        self.resnet = ResidualEncoder(6016, n_stages=5,
                                    features_per_stage=(2048, 1024, 512, 256, 128), conv_op=conv_op,
                                    kernel_sizes=5, strides=(1, 2, 2, 2, 2),
                                    n_blocks_per_stage=(2, 2, 2, 2, 2), conv_bias=True,
                                    norm_op=norm_op, norm_op_kwargs=None, dropout_op=nn.Dropout1d,
                                    dropout_op_kwargs={'p':0.2, 'inplace': True}, nonlin=nn.LeakyReLU,
                                    nonlin_kwargs={'inplace': True}, block=BasicBlockD,
                                    bottleneck_channels=None, return_skips=False,
                                    disable_default_stem=True,
                                    stem_channels=None,
                                    stochastic_depth_p=0.0,
                                    squeeze_excitation=False,
                                    squeeze_excitation_reduction_ratio=1/16)
        self.gap = get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type='avg')(1)
        self.input_dim = input_dim
                       
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True  # Makes input format [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, 2048, 8] (features over timesteps)
        Output: Tensor of shape [batch_size, 8, num_classes]
        """
        batch_size, num_features, num_timesteps = x.shape

        # Reshape to [batch_size * timesteps, 1, 2048, 1] for ResNet input
        #x = x.permute(0, 2, 1).unsqueeze(2)  # Shape: [batch_size, 8, 1, 2048]
        #x = x.reshape(-1, 1, self.input_dim)  # [batch_size * 8, 1, 2048, 1]

        #print(x.shape)

        # Extract features using ResNet
        features = self.resnet(x)  # Output shape: [batch_size * 8, 512, 1, 1]
        features = features.view(features.shape[0], -1)  # Flatten: [batch_size * 8, 256]

        # Reshape back to [batch_size, timesteps, 512] for Transformer
        #features = features.view(batch_size, num_timesteps, -1)

        # Pass through Transformer Encoder
        #transformer_out = self.transformer_encoder(features)  # [batch_size, 8, 512]

        # Fully connected classification per timestep
        output = self.fc(features)  # [batch_size, 8, num_classes]

        return output

class UNetWithClassificationBranches(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 num_classes_classification_branch: Union[List[int], Tuple[int, ...]],
                 features_per_stage_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 kernel_sizes_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 strides_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 use_encoding_layers_classification_branch: bool = True,
                 use_decoding_layers_classification_branch: bool = False,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__(input_channels,n_stages,features_per_stage,conv_op,kernel_sizes,strides,n_conv_per_stage,num_classes,n_conv_per_stage_decoder,conv_bias,norm_op,norm_op_kwargs,dropout_op,dropout_op_kwargs,nonlin,nonlin_kwargs,deep_supervision,nonlin_first)
        
        self.decoder = UNetDecoderExt(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        self.classification_branches = nn.ModuleList()
        self.gaps = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.use_encoding_layers_classification_branch = use_encoding_layers_classification_branch
        self.use_decoding_layers_classification_branch = use_decoding_layers_classification_branch

        ndim = 0
        if use_encoding_layers_classification_branch:
            ndim += np.sum(features_per_stage)
        if use_decoding_layers_classification_branch:
            ndim += np.sum(features_per_stage)
        if use_decoding_layers_classification_branch and use_encoding_layers_classification_branch:
            ndim -= features_per_stage[-1]

        # print("Using encoding layers for classification branch: ", use_encoding_layers_classification_branch)
        # print("Using decoding layers for classification branch: ", use_decoding_layers_classification_branch)
        # print("Number of features for classification branch: ", ndim)

        n_branches = len(num_classes_classification_branch)
        for i in range(n_branches):
            self.classification_branches.append(ResidualEncoder(ndim, n_stages=len(features_per_stage_classification_branch[i]),
                                       features_per_stage=features_per_stage_classification_branch[i], conv_op=conv_op,
                                       kernel_sizes=kernel_sizes_classification_branch[i], strides=strides_classification_branch[i],
                                       n_blocks_per_stage=2, conv_bias=True,
                                       norm_op=norm_op, norm_op_kwargs=None, dropout_op=nn.Dropout1d,
                                       dropout_op_kwargs={'p':0.2, 'inplace': True}, nonlin=nn.LeakyReLU,
                                       nonlin_kwargs={'inplace': True}, block=BasicBlockD,
                                       bottleneck_channels=None, return_skips=False,
                                       disable_default_stem=True,
                                       stem_channels=None,
                                       stochastic_depth_p=0.0,
                                       squeeze_excitation=False,
                                       squeeze_excitation_reduction_ratio=1/16))
            
            
            #self.classification_branches.append(TransformerClassifier(input_dim=features_per_stage[-1], num_classes=2, num_timesteps=8, hidden_dim=512, num_heads=8, num_layers=2, conv_op=conv_op, norm_op=norm_op))

            self.gaps.append(get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type='avg')(1))
            #self.classifiers.append(nn.Linear(2048, num_classes_classification_branch[i], True))
            self.classifiers.append(nn.Linear(features_per_stage_classification_branch[i][-1], num_classes_classification_branch[i], True))

        self.only_encode = False

    def concatenate(self, skips):
        #skips = skips[::-1]
        res = torch.avg_pool1d(skips[0].detach(), 32)
        st = 32
        for skip in skips[1:]:
            st = st // 2
            #print(res.shape, torch.avg_pool1d(skip.detach(), st).shape)
            res = torch.cat((res, torch.avg_pool1d(skip.detach(), st)), 1)
        return res
        #for i in range(len(skips)):

    def set_only_encode(self, only_encode):
        self.only_encode = only_encode

    def forward(self, x):
        skips = self.encoder(x)
        ys = []
        if self.use_decoding_layers_classification_branch:
            outskips = self.decoder.embed(skips)
            outskips.insert(0, skips[-1])
            outskips = outskips[::-1]

        ys = []
        for i in range(len(self.classification_branches)):
        #for i in range(1):
            #y = torch.cat((skips[-1].detach(), torch.avg_pool1d(skips[0].detach(), 32)) ,1)
            if self.use_encoding_layers_classification_branch and self.use_decoding_layers_classification_branch:
                y1 = self.concatenate(skips)
                y2 = self.concatenate(outskips[:-1])
                y = torch.cat((y1, y2), 1)
            elif self.use_decoding_layers_classification_branch:
                y = self.concatenate(outskips)
            elif self.use_encoding_layers_classification_branch:
                y = self.concatenate(skips)
            #y = torch.cat((y1, y2), 1)
            #y = torch.avg_pool1d(y, 16)
            #print(y.shape)

            #y = self.concatenate(skips)
            #print(y.shape)
            y = self.classification_branches[i](y)
            #print(y.shape)
            y = self.gaps[i](y)
            #print(y.shape)
            y = y.view(y.size(0), -1)
            #print(y.shape)
            y = self.classifiers[i](y)
            #y = y.reshape(-1, y.shape[2])
            #print(y.shape)
            #print(y.shape)
            ys.append(y)

        if self.only_encode:
            return ys
        
        return self.decoder(skips), ys

    def embed(self, x):
        skips = self.encoder(x)
        #print(skips[0].shape, skips[1].shape, skips[2].shape, skips[3].shape, skips[4].shape, skips[-1].shape)
        #latent = self.concatenate(skips)
        latents = self.decoder.embed(skips)
        return latents

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = np.int64(0)
        output += self.encoder.compute_conv_feature_map_size(input_size)
        output += self.decoder.compute_conv_feature_map_size(input_size)
        for i in range(len(self.classification_branches)):
            output += self.classification_branches[i].compute_conv_feature_map_size(input_size)

        return output


    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

class UNet3pWithClassificationBranches(UNetWithClassificationBranches):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 num_classes_classification_branch: Union[List[int], Tuple[int, ...]],
                 features_per_stage_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 kernel_sizes_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 strides_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 use_encoding_layers_classification_branch: bool = True,
                 use_decoding_layers_classification_branch: bool = False,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__(input_channels,n_stages,features_per_stage,conv_op,kernel_sizes,strides,n_conv_per_stage,num_classes,n_conv_per_stage_decoder,num_classes_classification_branch,features_per_stage_classification_branch,kernel_sizes_classification_branch,strides_classification_branch,use_encoding_layers_classification_branch,use_decoding_layers_classification_branch,conv_bias,norm_op,norm_op_kwargs,dropout_op,dropout_op_kwargs,nonlin,nonlin_kwargs,deep_supervision,nonlin_first)
        
        self.decoder = UNetDecoder3p(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        ndim = 0
        if use_encoding_layers_classification_branch:
            ndim += np.sum(features_per_stage)
        if use_decoding_layers_classification_branch:
            ndim += self.decoder.features_decoder*(n_stages-1) + features_per_stage[-1]

        print("Using encoding layers for classification branch: ", use_encoding_layers_classification_branch)
        print("Using decoding layers for classification branch: ", use_decoding_layers_classification_branch)
        print("Number of features for classification branch: ", ndim)

        self.classification_branches = nn.ModuleList()
        n_branches = len(num_classes_classification_branch)
        for i in range(n_branches):
            self.classification_branches.append(ResidualEncoder(ndim, n_stages=len(features_per_stage_classification_branch[i]),
                                       features_per_stage=features_per_stage_classification_branch[i], conv_op=conv_op,
                                       kernel_sizes=kernel_sizes_classification_branch[i], strides=strides_classification_branch[i],
                                       n_blocks_per_stage=2, conv_bias=True,
                                       norm_op=norm_op, norm_op_kwargs=None, dropout_op=nn.Dropout1d,
                                       dropout_op_kwargs={'p':0.2, 'inplace': True}, nonlin=nn.LeakyReLU,
                                       nonlin_kwargs={'inplace': True}, block=BasicBlockD,
                                       bottleneck_channels=None, return_skips=False,
                                       disable_default_stem=True,
                                       stem_channels=None,
                                       stochastic_depth_p=0.0,
                                       squeeze_excitation=False,
                                       squeeze_excitation_reduction_ratio=1/16))

    def concatenate(self, skips):
        #skips = skips[::-1]
        res = torch.avg_pool1d(skips[0].detach(), 32)
        st = 32
        for skip in skips[1:]:
            st = st // 2
            #print(res.shape, torch.avg_pool1d(skip.detach(), st).shape)
            res = torch.cat((res, torch.avg_pool1d(skip.detach(), st)), 1)
        return res
        #for i in range(len(skips)):

    def set_only_encode(self, only_encode):
        self.only_encode = only_encode

    def forward(self, x):
        skips = self.encoder(x)
        if self.use_decoding_layers_classification_branch:
            outskips = self.decoder.embed(skips)
            #outskips.insert(0, skips[-1])
            outskips = outskips[::-1]
        #print(len(outskips))
        #print(outskips[0].shape, outskips[1].shape, outskips[2].shape, outskips[3].shape, outskips[-1].shape)
        ys = []
        for i in range(len(self.classification_branches)):
            #y = torch.cat((skips[-1].detach(), torch.avg_pool1d(skips[0].detach(), 32)) ,1)
            if self.use_encoding_layers_classification_branch and self.use_decoding_layers_classification_branch:
                y1 = self.concatenate(skips)
                y2 = self.concatenate(outskips[:-1])
                y = torch.cat((y1, y2), 1)
            elif self.use_decoding_layers_classification_branch:
                y = self.concatenate(outskips)
            elif self.use_encoding_layers_classification_branch:
                y = self.concatenate(skips)
            #y = torch.cat((y1, y2), 1)
            #y = torch.avg_pool1d(y, 16)
            #print(y.shape)

            #y = self.concatenate(skips)
            #print(y.shape)
            y = self.classification_branches[i](y)
            #print(y.shape)
            y = self.gaps[i](y)
            #print(y.shape)
            y = y.view(y.size(0), -1)
            #print(y.shape)
            y = self.classifiers[i](y)
            #y = y.reshape(-1, y.shape[2])
            #print(y.shape)
            #print(y.shape)
            ys.append(y)

        if self.only_encode:
            return ys
        
        return self.decoder(skips), ys

    def embed(self, x):
        skips = self.encoder(x)
        #print(skips[0].shape, skips[1].shape, skips[2].shape, skips[3].shape, skips[4].shape, skips[-1].shape)
        #latent = self.concatenate(skips)
        latents = self.decoder.embed(skips)
        return latents

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size) + np.sum([c.compute_conv_feature_map_size(input_size) for c in self.classification_branches])


    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

class StackedConvLSTMBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )
        self.lstm = nn.LSTM(input_size=output_channels[-1], hidden_size=output_channels[-1], num_layers=1, batch_first=True)

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        print(x.shape)
        x = self.convs(x)
        print(self.output_channels)
        print(x.shape)
        x = self.lstm(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        output += np.prod([self.output_channels, *size_after_stride], dtype=np.int64)

        return output

class PlainConvEncoderLSTM(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvLSTMBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output

class UNetDecoderLSTM(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(x)
            elif s == (len(self.stages) - 1):
                seg_outputs.append(x)
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class UNetLSTMWithClassificationBranches(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 num_classes_classification_branch: Union[List[int], Tuple[int, ...]],
                 features_per_stage_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 kernel_sizes_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 strides_classification_branch: Union[List[int], Tuple[int, ...], List[List[int]], List[Tuple[int, ...]]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoderLSTM(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        self.lstms = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for i in range(n_stages):
            self.lstms.append(nn.LSTM(input_size=features_per_stage[n_stages-i-1], hidden_size=features_per_stage[n_stages-i-1], num_layers=1, batch_first=True, bidirectional=True))
            self.seg_layers.append(nn.Conv1d(2*features_per_stage[n_stages-i-1], num_classes, 1, 1, 0, bias=True))

        self.classification_branches = nn.ModuleList()
        self.gaps = nn.ModuleList()
        self.classification_lstms = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        n_branches = len(num_classes_classification_branch)
        for i in range(n_branches):
            self.classification_branches.append(ResidualEncoder(np.sum(features_per_stage), n_stages=len(features_per_stage_classification_branch[i]),
                                       features_per_stage=features_per_stage_classification_branch[i], conv_op=conv_op,
                                       kernel_sizes=kernel_sizes_classification_branch[i], strides=strides_classification_branch[i],
                                       n_blocks_per_stage=2, conv_bias=True,
                                       norm_op=norm_op, norm_op_kwargs=None, dropout_op=nn.Dropout1d,
                                       dropout_op_kwargs={'p':0.2, 'inplace': True}, nonlin=nn.LeakyReLU,
                                       nonlin_kwargs={'inplace': True}, block=BasicBlockD,
                                       bottleneck_channels=None, return_skips=False,
                                       disable_default_stem=True,
                                       stem_channels=None,
                                       stochastic_depth_p=0.0,
                                       squeeze_excitation=False,
                                       squeeze_excitation_reduction_ratio=1/16))

            self.gaps.append(get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type='avg')(1))
            self.classification_lstms.append(nn.LSTM(input_size=features_per_stage_classification_branch[i][-1], hidden_size=features_per_stage_classification_branch[i][-1], num_layers=1, batch_first=True, bidirectional=True))
            self.classifiers.append(nn.Linear(2*features_per_stage_classification_branch[i][-1], num_classes_classification_branch[i], True))

        self.only_encode = False

    def concatenate(self, skips):
        res = torch.avg_pool1d(skips[0].detach(), 32)
        st = 32
        for skip in skips[1:]:
            st = st // 2
            res = torch.cat((res, torch.avg_pool1d(skip.detach(), st)), 1)
        return res

    def concatenate_final_output(self, chunks, size, overlap, signal_length):

        final_map = torch.zeros(chunks.shape[0], chunks.shape[1], signal_length, device=chunks.device)
        count = torch.zeros(chunks.shape[0], 1, signal_length, device=chunks.device)
        chunks = chunks.unfold(dimension=2, size=size, step=overlap)  # Shape: (batch, channels, num_chunks, window_size)

        # During segmentation map creation
        for i in range(chunks.shape[-2]):
            start = i * overlap
            end = start + size
            print(start, end)
            print(chunks[:, :, i, :].shape)
            final_map[:, :, start:end] += chunks[:, :, i, :]
            count[:, :, start:end] += 1

        # Normalize by count
        full_segmentation_map = final_map / count

        return full_segmentation_map

    def forward(self, x):

        batch_size, _, signal_length = x.size()
        window_size = 1024  # UNet window size
        
        # Divide signal into chunks of window_size
        num_chunks = signal_length // window_size
        step = window_size
        chunks = x.unfold(dimension=2, size=window_size, step=step)  # Shape: (batch, channels, num_chunks, window_size)
        chunks = chunks.permute(0, 2, 1, 3)  # Shape: (batch, num_chunks, channels, window_size)
        
        # Process each chunk through the UNet
        unet_outputs = []
        y_outputs = []
        num_chunks = chunks.size(1)

        for i in range(num_chunks):
            chunk = chunks[:, i, :, :]  # Shape: (batch, channels, window_size)
            skips = self.encoder(chunk)

            ys = []
            for k in range(len(self.classification_branches)):
                y = self.classification_branches[k](self.concatenate(skips))
                y = self.gaps[k](y)
                ys.append(y)
            y_outputs.append(ys)

            unet_output = self.decoder(skips) # Shape: (batch, unet_output_channels, window_size)
            unet_outputs.append(unet_output)    

        ys = []
        for i in range(len(self.classification_branches)):
            y = torch.cat([y_outputs[k][i] for k in range(num_chunks)], dim=2)

            y = y.permute(0, 2, 1)  # Shape: (batch, window_size, channels)
            y, _ = self.classification_lstms[i](y)
            y = self.classifiers[i](y)
            y = y.permute(0, 2, 1)  # Shape: (batch, channels, window_size)
            y = y.mean(2)
            #print(y.shape)
            ys.append(y)


        #for layer in range(len(unet_outputs[0])):
        # Concatenate outputs along the time dimension
        unet_outputs = torch.cat(unet_outputs, dim=2)  # Shape: (batch, unet_output_channels, signal_length)

        # Transpose for LSTM: (batch, signal_length, channels)
        lstm_input = unet_outputs.permute(0, 2, 1)

        # Pass through LSTM
        lstm_output, _ = self.lstms[-1](lstm_input)  # Shape: (batch, signal_length, 2 * lstm_hidden_size)
        
        # Transpose back to (batch, channels, signal_length)
        lstm_output = lstm_output.permute(0, 2, 1)

        # Final convolution to produce segmentation map
        output = self.seg_layers[-1](lstm_output)  # Shape: (batch, num_classes, signal_length)
        #print(output.shape)

        #output = self.concatenate_final_output(output, window_size, step, signal_length)
        
        return output, ys

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size) + np.sum([c.compute_conv_feature_map_size(input_size) for c in self.classification_branches])



    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == '__main__':
    data = torch.rand((1, 1, 2048))

    model = UNet3pWithClassificationBranches(input_channels=1, 
                                            n_stages=5, 
                                            features_per_stage=(32, 64, 128, 256, 512), 
                                            conv_op=nn.Conv1d, 
                                            kernel_sizes=5, 
                                            strides=(1, 2, 2, 2, 2), 
                                            n_conv_per_stage=(2, 2, 2, 2, 2), 
                                            num_classes=6,
                                            n_conv_per_stage_decoder=(2, 2, 2, 2), 
                                            conv_bias=False, 
                                            num_classes_classification_branch=(2, 2),
                                            features_per_stage_classification_branch=((256,), (64,)),
                                            kernel_sizes_classification_branch=(5, 5),
                                            strides_classification_branch=(1, 2),
                                            norm_op=nn.BatchNorm1d, 
                                            norm_op_kwargs=None, 
                                            dropout_op=None, 
                                            dropout_op_kwargs=None, 
                                            nonlin=nn.LeakyReLU, 
                                            deep_supervision=True)

    if True:
        output = model(data)
        #dot = make_dot(output, params=dict(model.named_parameters()))
        #dot.render("model_visualization", format="png")

    #print(model.compute_conv_feature_map_size(data.shape[2:]))
