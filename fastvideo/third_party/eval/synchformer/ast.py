import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

# importing modified version of AST
from fastvideo.third_party.eval.synchformer.hf_src.modeling_ast import (ASTConfig, ASTForAudioClassification)
from fastvideo.third_party.eval.synchformer.motionformer import (AveragePooling, BaseEncoderLayer,
                                               TemporalTransformerEncoderLayer)


class AST(torch.nn.Module):

    def __init__(
        self,
        extract_features: bool = False,
        feat_type: str = None,
        max_spec_t: int = None,
        factorize_freq_time: bool = None,
        agg_freq_module: str = None,
        agg_time_module: str = None,
        add_global_repr: bool = True,
        agg_segments_module: str = None,
        max_segments: int = None,
    ) -> None:
        '''
            extract_features: if True, then the model will return the features instead of head's output
            ckpt_path: is not a path to a ckpt file, but a name of a model from the HuggingFace model hub.
            feat_type: if extract_features is True, this parameter specifies the type of features to return
            max_spec_t: if specified, then the model (pos emb) will be patched to support this length of spec
            factorize_freq_time: if True, then the model will use a factorized freq/time aggregation
            agg_freq_module: if specified, then the model will use this module for freq aggregation
            agg_time_module: if specified, then the model will use this module for time aggregation
            add_global_repr: if True, adds a global representation to the features (aggregation on segments)
            agg_segments_module: if specified, then the model will use this module for segments aggregation
            max_segments: if specified, the initialization of PE in the global agg module will use this value.
                          This should correspond to the max number of segments per video (if None, 16 is used)
        '''
        super().__init__()
        self.extract_features = extract_features
        self.max_spec_t = max_spec_t
        self.max_segments = max_segments

        # depending on whether the feat extractor was pre-trained contrastively or not, we need to
        # load the state dict differently.

        # if ckpt is specified, then load the model from the HuggingFace model hub, otherwise init a new model
        self.config = ASTConfig()
        self.config.num_labels = 527  # 2 by default, audioset has 527 labels
        full_model = ASTForAudioClassification(self.config)
        # logging.info('Initialized AST from scratch with the AST AudioSet config')

        # feature extractor
        self.ast = full_model.audio_spectrogram_transformer

        if self.extract_features:
            # assign `feat_type` (use default if not specified)
            self.feat_type = 'last_hidden_state' if feat_type is None else feat_type
            # define adapters if needed
            self.factorize_freq_time = factorize_freq_time
            # avoiding code duplication (used only if agg_*_module is TransformerEncoderLayer)
            transf_enc_layer_kwargs = dict(
                d_model=self.config.hidden_size,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                activation=nn.GELU(),
                batch_first=True,
                dropout=self.config.attention_probs_dropout_prob,
                layer_norm_eps=1e-6,
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
                    self.temp_attn_agg = nn.Identity()
            # define a global aggregation layer (aggregarate over segments)
            self.add_global_repr = add_global_repr
            if add_global_repr:
                if agg_segments_module == 'TransformerEncoderLayer':
                    # we can reuse the same layer as for temporal factorization (B, dim_to_agg, D) -> (B, D)
                    # we need to add pos emb (PE) because previously we added the same PE for each segment
                    pos_max_len = max_segments if max_segments is not None else 16  # 16 = 10sec//0.64sec + 1
                    self.global_attn_agg = TemporalTransformerEncoderLayer(
                        add_pos_emb=True,
                        pos_emb_drop=self.config.hidden_dropout_prob,
                        pos_max_len=pos_max_len,
                        **transf_enc_layer_kwargs)
                elif agg_segments_module == 'AveragePooling':
                    self.global_attn_agg = AveragePooling(avg_pattern='B S D -> B D')
        else:
            self.classifier = full_model.classifier

        # AST.device fails with AttributeError. This is a workaround
        self.device = full_model.device

        # pre-trained on 12*101+2=1214 tokens, but we have less (e.g. 12*6+2=74)
        self.patch_position_emb()

        # print the number of parameters
        # logging.info(f'AST: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}')

    def forward(self,
                x: torch.Tensor,
                for_loop: bool = False,
                cont_mask: torch.Tensor = None,
                **ast_kwargs) -> torch.Tensor:
        '''
            x: (B, S, T, F) where S is number of segments, F is number of (mel) frequency bins,
            ast_kwargs: additional arguments for the AST model
            cont_mask: (B, S, T, F) where 0s are the values to be masked out
            if `for_loop=True`, we use a for loop to extract features for each segment separately.
            if `for_loop=False`, we extract features for all segments at once.
                Using the for loop is slower but more memory efficient, while using all segments at once
                is faster but more memory inefficient.
                Using for loop allows to control the memory footprint by varying the number of videos in a
                batch (batch size) rather than the number of segments in a video.
        '''
        B, S, T, F = x.shape

        if for_loop:
            assert cont_mask is None, 'cont_mask is not supported with for_loop=True'
            orig_shape_s = (B, 1, T, F)
            # NOTE: since x is (B, S, T, F), and forward_segments expects (BS, T, F).
            # (B, S, T, F)[:, s] is (B, T, F) or (BS, T, F) if S=1.
            x = torch.cat([
                self.forward_segments(x[:, s], orig_shape_s, **ast_kwargs).unsqueeze(1)
                for s in range(S)
            ],
                          dim=1)
        else:
            orig_shape = (B, S, T, F)
            x = x.view(B * S, T, F)
            if cont_mask is not None:
                cont_mask = cont_mask.reshape(B * S, T, F)
            # AST expects a tensor of shape (B*S, T, F).
            x = self.forward_segments(x, orig_shape=orig_shape, cont_mask=cont_mask, **ast_kwargs)
            # unpack the segments (using rest dimensions to support different shapes e.g. (BS, D) or (BS, t, D))
            x = x.view(B, S, *x.shape[1:])
        # x now is of shape (B, S, D) or (B, S, t, D) if `self.temp_attn_agg` is `Identity`

        global_x = None
        if self.extract_features and self.add_global_repr:  # lazy execution, throws AttributeError
            assert len(x.shape) == 3, f'Local representation should be (B, S, D) {x.shape}'
            global_x = self.global_attn_agg(x)  # (B, D)

        return x, global_x  # x is (B, S, ...), global_x is (B, D) or None

    def forward_segments(self, x, orig_shape: tuple, cont_mask: torch.Tensor = None, **ast_kwargs):
        '''x is (BS, T, F), where S is the number of segments; cont_mask is (BS, T, F): 0s to be masked out'''
        # 'pooler_output': (B, D); or 'last_hidden_state: (B, T, D) where T is [CLS, DISTILL, <tokens>]
        # x_mask is (B, T) where 0s are the values to be masked out
        x, x_mask = self.ast(x, cont_mask=cont_mask, **ast_kwargs)

        if self.extract_features:
            x = self.get_features_by_type(x)
            if self.factorize_freq_time:
                x = self.restore_freq_temp_dims(x, orig_shape)  # (BS, D, f, t) <- (B*S, T, D)
                if cont_mask is not None:
                    # duplicating the mask for the latent dimension (D) to be compatible with the next func
                    x_mask = x_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
                    x_mask = self.restore_freq_temp_dims(x_mask,
                                                         orig_shape)  # (BS, D, f, t) <- (B*S, T, D)
                    # again removing the latent
                    x_mask = x_mask[:, 0, :, :]
                else:
                    x_mask = None
                x = self.freq_attn_agg(x, x_mask)  # (BS, t, D)
                x = self.temp_attn_agg(x)  # (BS, D) or (BS, t, D) if self.temp_attn_agg is Identity
        else:
            x = x['pooler_output']
            x = self.classifier(x)
        return x

    def get_features_by_type(self, x: BaseModelOutputWithPooling) -> torch.Tensor:
        if self.feat_type == 'pooler_output':
            return x['pooler_output']  # (B, D)
        elif self.feat_type == 'CLS':
            return x['last_hidden_state'][:, 0, :]  # (B, D)
        elif self.feat_type == 'last_hidden_state':
            return x['last_hidden_state']  # (B, 2+T, D)
        elif self.feat_type == 'last_hidden_state_no_AUX':
            return x['last_hidden_state'][:, 2:, :]  # (B, T, D) removing CLS and distill tokens
        else:
            raise ValueError(f'Unknown feature type: {self.feat_type}')

    def restore_freq_temp_dims(self, feats, orig_shape: tuple):
        '''
            feats are of shape (B*S, T, D)
                where T = 2 + f * t (if feat_type == 'last_hidden_state')
                where T =     f * t (if feat_type == 'last_hidden_state_no_AUX')
            Our goal is to make them of shape (B*S, f, t, D) where f and t are dimensions after patching.
            From `self.ast.embeddings.patch_embeddings`, it follows that we could reshape feats:
                `feats.transpose(1, 2).view(B*S, D, f, t)`

            (Similar function is defined in for RGB features in `motionformer.py`)
        '''
        B, S, T, F = orig_shape
        D = self.config.hidden_size

        # num patches in each dimension
        f, t = self.ast.embeddings.get_shape(self.config)

        if self.feat_type == 'last_hidden_state':
            feats = feats[:, 2:, :]  # removing CLS and distill tokens

        feats = feats.permute(0, 2, 1)  # (B*S, D, T)
        feats = feats.view(B * S, D, f, t)  # (B*S, D, f, t)

        return feats

    def patch_position_emb(self):
        if self.max_spec_t is not None:
            self.config.max_length = self.max_spec_t
        f, t = self.ast.embeddings.get_shape(self.config)
        shortened = self.ast.embeddings.position_embeddings[:, :f * t + 2].clone(
        )  # +2 for CLS and distill tokens
        self.ast.embeddings.position_embeddings = torch.nn.Parameter(shortened).to(self.device)

    def to(self, device):
        '''AST.device fails with AttributeError. This is a workaround. '''
        self.device = torch.device(device)
        return super().to(device)


class FrequencyTransformerEncoderLayer(BaseEncoderLayer):
    ''' This layer is used to aggregate the features along the frequency axis.
    It follows the same logic as spatio-temporal aggregation in visual feature extractor.
    Thus, it is recommended to check the definition of `BaseEncoderLayer` in `motionformer.py` '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        ''' x: (B*S, D, f, t); if specified x_mask (B*S, f, t), 0s are the values to be masked out '''
        BS, D, f, t = x.shape

        # time as a batch dimension
        x = x.permute(0, 3, 2, 1)  # (B*S, t, f, D)
        x = x.reshape(BS * t, f, D)  # .view() fails with non-contiguous memory
        # similar to mask
        if x_mask is not None:
            x_mask = x_mask.permute(0, 2, 1)  # (B*S, t, f)
            x_mask = x_mask.reshape(BS * t, f)

        # apply encoder layer (BaseEncoderLayer.forward) - it will add CLS token and output its representation
        x = super().forward(x=x, x_mask=x_mask)  # (B*S*t, D)

        # reshape back to (B*S, t, D)
        x = x.view(BS, t, D)

        return x  # (B*S, t, D)
