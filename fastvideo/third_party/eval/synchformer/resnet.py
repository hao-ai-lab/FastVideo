import sys
from pathlib import Path
import logging
import einops

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

sys.path.append('.')  # nopep8

from utils.utils import check_if_file_exists_else_download
from model.modules.feat_extractors.audio.ast import FrequencyTransformerEncoderLayer
from model.modules.feat_extractors.visual.motionformer import AveragePooling, TemporalTransformerEncoderLayer


class ResNetAudio(ResNet):

    def __init__(self, arch_name, num_classes, extract_features, ckpt_path=None, **kwargs):

        if arch_name == 'resnet18':
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif arch_name == 'resnet34':
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif arch_name == 'resnet50':
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif arch_name == 'resnet101':
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif arch_name == 'resnet152':
            block = Bottleneck
            layers = [3, 8, 36, 3]
        else:
            raise NotImplementedError

        super().__init__(block, layers, num_classes, **kwargs)

        # replacing the old conv1 to the new one (RGB - 3; spectrogram - 1)
        conv1 = self.conv1
        self.conv1 = torch.nn.Conv2d(1, conv1.out_channels, conv1.kernel_size,
                                     conv1.stride, conv1.padding, bias=conv1.bias)
        self.extract_features = extract_features
        self.embed_dim = self.fc.in_features

        # load the ckpt
        load_state_dict_resnet(self, ckpt_path, prefix='afeat_extractor.')

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.extract_features:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return super().forward(x)


class ResNet18AudioFeatures(ResNetAudio):

    # ckpt_path should default to None, otherwise when no pre-training is desired it will throw an error
    def __init__(self,
                 extract_features: bool = False,
                 ckpt_path: str = None,
                 feat_type: str = None,
                 max_spec_t: int = None,
                 factorize_freq_time: bool = None,
                 agg_freq_module: str = None,
                 agg_time_module: str = None,
                 add_global_repr: bool = True,
                 agg_segments_module: str = None,
                 max_segments: int = None,
                 ) -> None:
        super().__init__(arch_name='resnet18', num_classes=308, extract_features=extract_features,
                         ckpt_path=ckpt_path)
        assert extract_features, 'Not implemented otherwise'
        self.extract_features = extract_features
        self.feat_type = feat_type
        self.max_spec_t = max_spec_t
        # similar to s3d
        self.nhead = 8
        self.mlp_ratio = 4
        self.drop_rate = 0.0

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            was_pt_on_vgs_cls = 'ResNetAudio-' in Path(ckpt_path).stem
            if was_pt_on_vgs_cls:
                self.load_state_dict(ckpt['model'], strict=True)

        # saving some memory
        if extract_features:
            self.avgpool = torch.nn.Identity()
            self.fc = torch.nn.Identity()

        # define adapters if needed
        self.factorize_freq_time = factorize_freq_time
        # avoiding code duplication (used only if agg_*_module is TransformerEncoderLayer)
        transf_enc_layer_kwargs = dict(
            d_model=self.embed_dim, nhead=self.nhead, dim_feedforward=self.mlp_ratio*self.embed_dim,
            activation=torch.nn.GELU(), batch_first=True, dropout=self.drop_rate, layer_norm_eps=1e-6,
            norm_first=True,
        )
        if factorize_freq_time:
            self.feat_type = 'last_hidden_state'  # this feat_type supports factorization
            # frequency aggreration
            if agg_freq_module == 'TransformerEncoderLayer':
                self.freq_attn_agg = FrequencyTransformerEncoderLayer(**transf_enc_layer_kwargs)
            elif agg_freq_module == 'AveragePooling':
                self.freq_attn_agg = AveragePooling(avg_pattern='BS D f t -> BS D t',
                                                    then_permute_pattern='BS D t -> BS t D')
            # time aggreration
            if agg_time_module == 'TransformerEncoderLayer':
                self.temp_attn_agg = TemporalTransformerEncoderLayer(**transf_enc_layer_kwargs)
            elif agg_time_module == 'AveragePooling':
                self.temp_attn_agg = AveragePooling(avg_pattern='BS t D -> BS D')
            elif 'Identity' in agg_time_module:
                self.temp_attn_agg = torch.nn.Identity()
        # define a global aggregation layer (aggregarate over segments)
        self.add_global_repr = add_global_repr
        if add_global_repr:
            if agg_segments_module == 'TransformerEncoderLayer':
                # we can reuse the same layer as for temporal factorization (B, dim_to_agg, D) -> (B, D)
                # we need to add pos emb (PE) because previously we added the same PE for each segment
                pos_max_len = max_segments if max_segments is not None else 16  # 16 = 10sec//0.64sec + 1
                self.global_attn_agg = TemporalTransformerEncoderLayer(
                    add_pos_emb=True, pos_emb_drop=self.drop_rate,
                    pos_max_len=pos_max_len, **transf_enc_layer_kwargs
                )
            elif agg_segments_module == 'AveragePooling':
                self.global_attn_agg = AveragePooling(avg_pattern='B S D -> B D')

        # do not keep fc to save memory
        self.fc = torch.nn.Identity()

        if ckpt_path is not None:
            ckpt = ckpt['state_dict']
            was_pt_on_avclip = any('a_encoder.' in k[0] or 'v_encoder.' in k[0] for k in ckpt.items())
            assert was_pt_on_vgs_cls is False, f'Unexpected ckpt: {ckpt_path}'
            if was_pt_on_avclip:
                ckpt_weights = dict()
                for k, v in ckpt.items():
                    if k.startswith(('module.a_encoder.', 'a_encoder.')):
                        k = k.replace('module.', '').replace('a_encoder.', '')
                        ckpt_weights[k] = v
                _load_status = self.load_state_dict(ckpt_weights, strict=False)
                if len(_load_status.missing_keys) > 0 or len(_load_status.unexpected_keys) > 0:
                    logging.warning(f'Loading exact ckpt from {ckpt_path} failed. \n' \
                                    f'Missing keys ({len(_load_status.missing_keys)}): ' \
                                    f'{_load_status.missing_keys}, \n' \
                                    f'Unexpected keys ({len(_load_status.unexpected_keys)}): ' \
                                    f'{_load_status.unexpected_keys} \n' \
                                    f'freq_attn_agg are expected to be unexpected if ckpt was pt contrastively '\
                                    f'as well as fc could be missing because we use features, not a classifier.')
                else:
                    logging.info(f'Loading ResNet ckpt from {ckpt_path} succeeded.')

        # print the number of parameters
        logging.info(f'afeat_extractor: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}')

    def forward(self, x: torch.Tensor, for_loop: bool = False, cont_mask: torch.Tensor = None):
        assert for_loop is False and cont_mask is None, 'Not implemented'
        B, S, T, F = x.shape

        # (BS, D) <- (B, S, T, D)
        x = self.forward_segments(x)

        # unpack the segments (using rest dimensions to support different shapes e.g. (BS, D) or (BS, t, D))
        x = x.view(B, S, *x.shape[1:])
        # x now is of shape (B, S, D) or (B, S, t, D) if `self.temp_attn_agg` is `Identity`

        global_x = None
        if self.extract_features and self.add_global_repr:  # lazy execution, throws AttributeError
            assert len(x.shape) == 3, f'Local representation should be (B, S, D) {x.shape}'
            global_x = self.global_attn_agg(x)  # (B, D)

        return x, global_x  # x is (B, S, ...), global_x is (B, D) or None

    def forward_segments(self, x):
        x = einops.rearrange(x, 'B S T F -> (B S) 1 F T')
        # (BS, D, f, t) <- (BS, 1, F, T)
        x = super().forward(x)

        if self.extract_features:
            if self.factorize_freq_time:
                x = self.freq_attn_agg(x)  # (BS, t, D)
                x = self.temp_attn_agg(x)  # (BS, D) or (BS, t, D) if self.temp_attn_agg is Identity
        return x


def load_state_dict_resnet(model, ckpt_path, prefix):
    if ckpt_path is not None:
        check_if_file_exists_else_download(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        ckpt = ckpt.get('model', ckpt.get('state_dict', ckpt))
        # we need to filter out the state_dict of the AVCLIP model (has both A and V extractors)
        # and keep only the state_dict of the feat extractor
        # FIXME: this is a bit hacky, but it works
        was_pt_on_avclip = any('a_encoder.' in k[0] or 'v_encoder.' in k[0] for k in ckpt.items())
        if not was_pt_on_avclip:
            model.load_state_dict(ckpt)
            logging.info(f'Loading ResNet ckpt from {ckpt_path} succeeded.')
        # if was_pt_on_avclip:
        #     ckpt_weights = dict()
        #     for k, v in ckpt.items():
        #         if k.startswith(('module.a_encoder.', 'a_encoder.')):
        #             k = k.replace('module.', '').replace('a_encoder.', '')
        #             ckpt_weights[k] = v
        #     _load_status = model.load_state_dict(ckpt_weights, strict=False)
        #     if len(_load_status.missing_keys) > 0 or len(_load_status.unexpected_keys) > 0:
        #         logging.warning(f'Loading exact ckpt from {ckpt_path} failed. \n' \
        #                         f'Missing keys ({len(_load_status.missing_keys)}): ' \
        #                         f'{_load_status.missing_keys}, \n' \
        #                         f'Unexpected keys ({len(_load_status.unexpected_keys)}): ' \
        #                         f'{_load_status.unexpected_keys} \n' \
        #                         f'freq_attn_agg are expected to be unexpected if ckpt was pt contrastively '\
        #                         f'as well as fc could be missing because we use features, not a classifier.')
        #     else:
        #         logging.info(f'Loading ResNet ckpt from {ckpt_path} succeeded.')
        # else:
        #     model.load_state_dict(ckpt)
        #     logging.info(f'Loading ResNet ckpt from {ckpt_path} succeeded.')


if __name__ == '__main__':
    B = 2
    ckpt_path = './model/modules/feat_extractors/audio/22-06-24T08-10-33/ResNetAudio-22-06-24T08-10-33.pt'
    afeat_extractor = ResNet18AudioFeatures(ckpt_path=ckpt_path)
    # x = torch.rand(B, 1, 257, 1551)
    # x = torch.rand(B, 1, 257, 1379)
    x = torch.rand(B, 1, 128, 66)
    x = afeat_extractor(x)
    print(x.shape)
