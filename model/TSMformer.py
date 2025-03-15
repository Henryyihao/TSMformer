import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from utils.RevIN import RevIN
from layers.Transformer_EncDec import Encoder, FeatureGroupTemporalEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        self.revin = RevIN(configs.enc_in)
        self.use_norm = configs.use_norm
        self.decomp = series_decomp(configs.moving_avg)

        # seasonal Linear encoder
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(configs.seq_len, 4 * configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(4 * configs.d_model, configs.d_model)
        )

        # Trend embedding
        self.trend_embedding = nn.Linear(configs.seq_len, configs.d_model)

        # Trend（Specialized Transformer Encoder）
        self.trend_encoder = Encoder(
            [
                FeatureGroupTemporalEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    nums_feature=configs.enc_in,
                    d_ff=configs.d_ff,  # 减小FFN维度
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Dual predictor heads
        self.linear_head = nn.Linear(configs.d_model, configs.pred_len)
        self.mlp_head = nn.Sequential(
            nn.Linear(configs.d_model, 4 * configs.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * configs.d_model, 2 * configs.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * configs.d_model, configs.pred_len)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            x_enc = self.revin(x_enc, 'norm')  # [B, L, N]

        trend, seasonal = self.decomp(x_enc)

        trend_embed = self.trend_embedding(trend.permute(0, 2, 1))
        trend_feat, _ = self.trend_encoder(trend_embed)  # [B, N, D]

        seasonal_feat = self.seasonal_encoder(seasonal.permute(0, 2, 1))  # [B, N, D]

        fused = trend_feat + seasonal_feat
        linear_out = self.linear_head(fused)
        mlp_out = self.mlp_head(fused)  # [B, N, L]
        out = (linear_out + mlp_out).permute(0, 2, 1)
        if self.use_norm:
            out = self.revin(out, 'denorm')
        return out