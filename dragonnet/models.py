import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import PAD_VALUE, N_FEATS_PER_PLAYER_PER_TIMESTEP, N_GEOMETRIC_FEATS, N_OFFENSE, N_DEFENSE

class OutcomeRegressor(nn.Module):
    def __init__(self, shared_dim, outcome_hidden):
        super(OutcomeRegressor, self).__init__()
        self.fc1 = nn.Linear(in_features=shared_dim, out_features=outcome_hidden)
        self.fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.fc_out = nn.Linear(in_features=outcome_hidden, out_features=1)

    def forward(self, Z):
        yout = F.relu(self.fc1(Z))
        yout = F.relu(self.fc2(yout))
        yout = self.fc_out(yout)
        return yout.squeeze(-1)

class PropensityModel(nn.Module):
    def __init__(self, shared_dim):
        super(PropensityModel, self).__init__()
        self.treat_out = nn.Linear(in_features=shared_dim, out_features=1)
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)  # this is just a trainable scalar parameter used for the targeted regularization

    def forward(self, Z):
        t_pred = torch.sigmoid(self.treat_out(Z))
        eps = self.epsilon(torch.ones(t_pred.size(0), device=self.epsilon.weight.device).unsqueeze(-1))
        return t_pred.squeeze(-1), eps.squeeze(-1)

class NonLinearPropensityModel(nn.Module):
    def __init__(self, shared_dim, hidden_dim=100):
        super(NonLinearPropensityModel, self).__init__()
        self.hidden = nn.Linear(in_features=shared_dim, out_features=hidden_dim)
        self.treat_out = nn.Linear(in_features=hidden_dim, out_features=1)
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)  # this is just a trainable scalar parameter used for the targeted regularization

    def forward(self, Z):
        out = self.hidden(Z)
        out = self.treat_out(out)
        t_pred = torch.sigmoid(out)
        eps = self.epsilon(torch.ones(t_pred.size(0), device=self.epsilon.weight.device).unsqueeze(-1))
        return t_pred.squeeze(-1), eps.squeeze(-1)


class TransformerRepresentor(nn.Module):
    def __init__(self, input_dim, embed_dim=128, n_attention_heads=8, n_encoder_layers=3):
        super(TransformerRepresentor, self).__init__()
        self.positional_encoder = PositionalEncoding(embed_dim)
        self.linear_embed = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_attention_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.output_size = embed_dim

    def forward(self, X):
        X = X.float()
        mask = (X[..., 0] == PAD_VALUE)
        out = self.linear_embed(X)
        out = self.positional_encoder(out)
        out = self.transformer_encoder(out, src_key_padding_mask=mask)  # dim is (batch_size, time, features)
        out_pooled, _ = torch.max(out * ~mask.unsqueeze(-1), dim=1)
        return out_pooled

class TransformerRepresentorWithPlayContext(TransformerRepresentor):
    def __init__(self, input_dims, play_context_embed_dim=128, cross_attention_heads=8, **kwargs):
        self.input_dims = input_dims
        time_series_input_dim, play_context_input_dim = input_dims

        super(TransformerRepresentorWithPlayContext, self).__init__(time_series_input_dim, **kwargs)
        self.play_context_embed = nn.Linear(play_context_input_dim, play_context_embed_dim)
        self.xatt = nn.MultiheadAttention(self.output_size, cross_attention_heads, kdim=None, vdim=play_context_embed_dim, batch_first=True)

    def forward(self, X):
        X_timeseries, X_static = X
        X_timeseries = X_timeseries.float()
        X_static = X_static.float()

        mask = (X[..., 0] == PAD_VALUE)
        out = self.positional_encoder(X)
        out = self.linear_embed(X)
        out = self.transformer_encoder(out, src_key_padding_mask=mask)  # dim is (batch_size, time, features)
        # here, we fuse in static play data -- in collate_fn, should be repeated, then padded
        attn_mask = (X_static[..., 0] == PAD_VALUE)
        static_out = self.static_embed(X_static)  # (batch_size, time, features')
        # call a MHA with static play features as the values (batch_size, time  [tiled], features) and pass in attn_mask
        final_out = self.xatt(out, out, static_out, src_key_padding_mask=mask, attn_mask=attn_mask)

        out_pooled, _ = torch.max(final_out * ~mask.unsqueeze(-1), dim=1)
        return out_pooled

class SimplifiedMultiLevelTransformer(nn.Module):
    def __init__(
        self,
        input_dims,
        geom_embed_dim=128,
        player_embed_dim=16,
        geom_n_attention_heads=8,
        geom_n_encoder_layers=3,
        ball_carrier_n_attention_heads=8,
        ball_carrier_n_encoder_layers=3,
        drop_absolute_x=False,
        drop_absolute_x_from_all=False,
    ):
        super(SimplifiedMultiLevelTransformer, self).__init__()
        geom_input_dim, ball_carrier_input_dim = input_dims
        self.geom_embed_dim = geom_embed_dim
        self.player_embed_dim = player_embed_dim
        self.input_dims = input_dims
        self.geometric_transformer = TransformerRepresentor(
            geom_input_dim,
            embed_dim=geom_embed_dim,
            n_attention_heads=geom_n_attention_heads,
            n_encoder_layers=geom_n_encoder_layers
        )

        self.ball_carrier_transformer = TransformerRepresentor(
            ball_carrier_input_dim,
            embed_dim=player_embed_dim,
            n_attention_heads=ball_carrier_n_attention_heads,
            n_encoder_layers=ball_carrier_n_encoder_layers
        )
        self.drop_absolute_x = drop_absolute_x
        self.drop_absolute_x_from_all = drop_absolute_x_from_all
        self.output_size = geom_embed_dim + player_embed_dim

    def forward(self, batch):
        if len(batch) == 4:  # HACK: for backward compatibility with old collate functions
            X_geometric, X_ball_carrier, _, _ = batch
        else:
            X_geometric, X_ball_carrier = batch
        X_geometric = X_geometric.float()
        X_ball_carrier = X_ball_carrier.float()
        if self.drop_absolute_x_from_all:
            n_feats = X_geometric.size(2)
            raw_feature_start_idx = (N_OFFENSE + N_DEFENSE) * N_GEOMETRIC_FEATS
            new_indices = (torch.arange(n_feats) < raw_feature_start_idx) | (torch.remainder(torch.arange(n_feats) - raw_feature_start_idx, N_FEATS_PER_PLAYER_PER_TIMESTEP) != 0)
            X_geometric = X_geometric[:, :, new_indices]
        if self.drop_absolute_x:
            X_ball_carrier = X_ball_carrier[:, :, 1:]
        geometric_out = self.geometric_transformer(X_geometric)
        ball_carrier_out = self.ball_carrier_transformer(X_ball_carrier)
        final_out = torch.cat([ball_carrier_out, geometric_out], dim=1)
        return final_out

class MultiLevelTransformerRepresentor(nn.Module):
    def __init__(self,
                 input_dims,
                 geom_embed_dim=128,
                 player_embed_dim=16,
                 geom_n_attention_heads=8,
                 geom_n_encoder_layers=3,
                 ball_carrier_n_attention_heads=8,
                 ball_carrier_n_encoder_layers=3,
                 tackler_n_attention_heads=8,
                 tackler_n_encoder_layers=3,
                 ):
        super(MultiLevelTransformerRepresentor, self).__init__()
        geom_input_dim, ball_carrier_input_dim, tackler_input_dim = input_dims
        self.geom_embed_dim = geom_embed_dim
        self.player_embed_dim = player_embed_dim
        self.input_dims = input_dims
        self.geometric_transformer = TransformerRepresentor(
            geom_input_dim,
            embed_dim=geom_embed_dim,
            n_attention_heads=geom_n_attention_heads,
            n_encoder_layers=geom_n_encoder_layers
        )

        self.ball_carrier_transformer = TransformerRepresentor(
            ball_carrier_input_dim,
            embed_dim=player_embed_dim,
            n_attention_heads=ball_carrier_n_attention_heads,
            n_encoder_layers=ball_carrier_n_encoder_layers
        )

        self.tackler_transformer = TransformerRepresentor(
            tackler_input_dim,
            embed_dim=player_embed_dim,
            n_attention_heads=tackler_n_attention_heads,
            n_encoder_layers=tackler_n_encoder_layers
        )

        self.output_size = geom_embed_dim + player_embed_dim

    def forward(self, batch):
        X_geometric, X_ball_carrier, X_tacklers, n_tacklers = batch
        X_geometric = X_geometric.float()
        X_ball_carrier = X_ball_carrier.float()
        X_tacklers = X_tacklers.float()
        n_tacklers = n_tacklers.detach().clone()

        geometric_out = self.geometric_transformer(X_geometric)
        ball_carrier_out = self.ball_carrier_transformer(X_ball_carrier)

        tackler_out = []
        for i, X_single_tackler in enumerate(torch.split(X_tacklers, 1, dim=2)):  # (batch_size, time, max_tacklers_in_batch, n_feats)
            single_tackler_out = self.tackler_transformer(X_single_tackler.squeeze(2))  # (batch_size, time, embed_dim)
            tackler_out.append(single_tackler_out)
        tackler_mask = F.one_hot(n_tacklers)[:, 1:]
        final_tackler_mask = (tackler_mask - torch.cumsum(tackler_mask, dim=1) + 1).bool().unsqueeze(1).repeat(1, self.player_embed_dim, 1)  # create a mask using one-hot manipulation
        tackler_out = torch.stack(tackler_out, dim=2)
        tacklers_masked = torch.masked.masked_tensor(tackler_out, final_tackler_mask)
        tacklers_masked_agg = tacklers_masked.sum(dim=2)

        assert torch.all(tacklers_masked_agg.get_mask()).item()
        absolute_out = tacklers_masked_agg.get_data() + ball_carrier_out  # using .get_data() directly on a masked tensor directly isn't recommended but in this case we know .sum() should reduce them out
        final_out = torch.cat([absolute_out, geometric_out], dim=1)

        return final_out


class PositionalEncoding(nn.Module):
    """
        Adapted from the Pytorch tutorial on transformers.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # (max_len, d_model // 2)
        if d_model % 2:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])  # (max_len, d_model // 2 - 1) handle odd feature dimensionalities
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)  # (max_len, d_model // 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderLayerWithCrossAttention(nn.TransformerEncoderLayer):
    """
        Based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        without the checks for usage of FlashAttention.
    """
    def forward(self, src, ctx, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._xa_block(self.norm1(x), ctx, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._xa_block(x, ctx, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _xa_block(self, x, ctx, attn_mask, key_padding_mask, is_causal=False):
        x = self.self_attn(ctx, x, x,  # q = ctx, kv = x
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _sa_block(self, *args, **kwargs):
        raise RuntimeError(f"Since this is an instance of {self.__class__.__name__}, cross-attention should be used. Did you mean to use TransformerEncoderLayer?")

class DragonNet(nn.Module):
    """
    Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for covariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dims, backbone_class, outcome_hidden=100, nonlinear_propensity_model=False, **kwargs):
        super(DragonNet, self).__init__()
        self.backbone = backbone_class(input_dims, **kwargs)
        self.shared_dim = self.backbone.output_size
        if nonlinear_propensity_model:
            self.propensity_model = NonLinearPropensityModel(self.shared_dim)
        else:
            self.propensity_model = PropensityModel(self.shared_dim)
        self.y0_model = OutcomeRegressor(self.shared_dim, outcome_hidden)
        self.y1_model = OutcomeRegressor(self.shared_dim, outcome_hidden)

    def forward(self, inputs):
        z = self.backbone(inputs)
        y0 = self.y0_model(z)
        y1 = self.y1_model(z)
        t_pred, eps = self.propensity_model(z)

        return y0, y1, t_pred, eps
