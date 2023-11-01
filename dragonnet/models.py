import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import PAD_VALUE

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

class TransformerRepresentor(nn.Module):
    def __init__(self, input_dim, n_attention_heads=8, n_encoder_layers=3):
        super(TransformerRepresentor, self).__init__()
        self.positional_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_attention_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.output_size = input_dim

    def forward(self, X):
        mask = (X[..., 0] == PAD_VALUE)
        out = self.positional_encoder(X)
        out = self.transformer_encoder(out, src_key_padding_mask=mask)  # dim is (batch_size, time, features)
        out_pooled, _ = torch.max(out * ~mask.unsqueeze(-1), dim=1)  # TODO: mask this correctly
        return out_pooled

class PositionalEncoding(nn.Module):
    """
        Adapted from the Pytorch tutorial on transformers.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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
    def __init__(self, input_dim, backbone_class, outcome_hidden=100, **kwargs):
        super(DragonNet, self).__init__()
        self.backbone = backbone_class(input_dim, **kwargs)
        self.shared_dim = self.backbone.output_size
        self.propensity_model = PropensityModel(self.shared_dim)
        self.y0_model = OutcomeRegressor(self.shared_dim, outcome_hidden)
        self.y1_model = OutcomeRegressor(self.shared_dim, outcome_hidden)

    def forward(self, inputs):
        z = self.backbone(inputs)
        y0 = self.y0_model(z)
        y1 = self.y1_model(z)
        t_pred, eps = self.propensity_model(z)

        return y0, y1, t_pred, eps
