import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# class DualStreamEncoderLayer(nn.Module):
#     def __init__(self, attention1, attention2, d_model, d_ff=None, dropout=0.1, activation="relu", ema_alpha=0.1):
#         super(DualStreamEncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention1 = attention1  # Attention for the original stream
#         self.attention2 = attention2  # Attention for the smoothed stream
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model // 2)  # Corrected: Normalize output of attention1
#         self.norm2 = nn.LayerNorm(d_model // 2)  # Corrected: Normalize output of attention2
#         self.norm_ff = nn.LayerNorm(d_model) # Normalize after fusion and feedforward
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.ema_alpha = ema_alpha
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # Stream 1: Original input
#         new_x1, attn1 = self.attention1(
#             x[:, :, :self.attention1.query_projection.out_features],
#             x[:, :, :self.attention1.key_projection.out_features],
#             x[:, :, :self.attention1.value_projection.out_features],
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x1 = x[:, :, :self.attention1.out_projection.out_features] + self.dropout(new_x1)
#         x1 = self.norm1(x1)
#
#         # Stream 2: Exponentially smoothed input
#         x_smooth = self._exponential_smoothing(x, alpha=self.ema_alpha)
#         new_x2, attn2 = self.attention2(
#             x_smooth[:, :, :self.attention2.query_projection.out_features],
#             x_smooth[:, :, :self.attention2.key_projection.out_features],
#             x_smooth[:, :, :self.attention2.value_projection.out_features],
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x2 = x_smooth[:, :, :self.attention2.out_projection.out_features] + self.dropout(new_x2)
#         x2 = self.norm2(x2)
#
#         # Fusion (Concatenation and Linear Projection)
#         y = torch.cat([x1, x2], dim=-1)
#
#         y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))
#
#         return self.norm_ff(y + y_ff), (attn1, attn2)
#
#     def _exponential_smoothing(self, x, alpha):
#         # x: [B, N, E]
#         B, N, E = x.shape
#         smoothed = torch.zeros_like(x).to(x.device) # Ensure smoothed is on the same device
#         smoothed[:, 0, :] = x[:, 0, :]
#         for t in range(1, N):
#             smoothed[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * smoothed[:, t - 1, :]
#         return smoothed


# class DualStreamEncoderLayer(nn.Module):
#     def __init__(self, attention1, attention2, d_model, d_ff=None, dropout=0.1, activation="relu", ema_alpha=0.1):
#         super(DualStreamEncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention1 = attention1  # Attention for the original stream
#         self.attention2 = attention2  # Attention for the residual stream
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model // 2)
#         self.norm2 = nn.LayerNorm(d_model // 2)
#         self.norm_ff = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.ema = EMA(ema_alpha) # Use the efficient EMA implementation
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # Stream 1: Original input
#         new_x1, attn1 = self.attention1(
#             x[:, :, :self.attention1.query_projection.out_features],
#             x[:, :, :self.attention1.key_projection.out_features],
#             x[:, :, :self.attention1.value_projection.out_features],
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x1 = x[:, :, :self.attention1.out_projection.out_features] + self.dropout(new_x1)
#         x1 = self.norm1(x1)
#
#         # Stream 2: Residual (Original - Trend)
#         trend = self.ema(x.transpose(1, 2)).transpose(1, 2) # Apply EMA along the time dimension
#         residual = x - trend
#         new_x2, attn2 = self.attention2(
#             residual[:, :, :self.attention2.query_projection.out_features],
#             residual[:, :, :self.attention2.key_projection.out_features],
#             residual[:, :, :self.attention2.value_projection.out_features],
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x2 = residual[:, :, :self.attention2.out_projection.out_features] + self.dropout(new_x2)
#         x2 = self.norm2(x2)
#
#         # Fusion (Concatenation and Linear Projection)
#         y = torch.cat([x1, x2], dim=-1)
#
#         y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))
#
#         return self.norm_ff(y + y_ff), (attn1, attn2)


class DualStreamEncoderLayer(nn.Module):
    def __init__(self, attention1, attention2, d_model, d_ff=None, dropout=0.1, activation="relu", holt_alpha=0.1, holt_beta=0.1):
        super(DualStreamEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention1 = attention1  # Attention for the original stream
        self.attention2 = attention2  # Attention for the smoothed stream
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 2)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.holt_alpha = holt_alpha
        self.holt_beta = holt_beta


    def _double_exponential_smoothing(self, x, alpha, beta):
        B, N, E = x.shape
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        beta = torch.tensor(beta, dtype=x.dtype, device=x.device)

        # 初始化水平和趋势
        level = x[:, 0:1, :].clone()
        trend = torch.zeros_like(level)

        levels = [level]
        trends = [trend]

        for t in range(1, N):
            new_level = alpha * x[:, t:t + 1, :] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend

            levels.append(new_level)
            trends.append(new_trend)

            level, trend = new_level, new_trend

        return torch.cat(levels, dim=1)


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Stream 1: Original input
        new_x1, attn1 = self.attention1(
            x[:, :, :self.attention1.query_projection.out_features],
            x[:, :, :self.attention1.key_projection.out_features],
            x[:, :, :self.attention1.value_projection.out_features],
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x1 = x[:, :, :self.attention1.out_projection.out_features] + self.dropout(new_x1)
        x1 = self.norm1(x1)

        # Stream 2: Double Exponentially Smoothed input
        x_smooth = self._double_exponential_smoothing(x, float(self.holt_alpha), float(self.holt_beta))
        new_x2, attn2 = self.attention2(
            x_smooth[:, :, :self.attention2.query_projection.out_features],
            x_smooth[:, :, :self.attention2.key_projection.out_features],
            x_smooth[:, :, :self.attention2.value_projection.out_features],
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x2 = x_smooth[:, :, :self.attention2.out_projection.out_features] + self.dropout(new_x2)
        x2 = self.norm2(x2)

        # Fusion (Concatenation and Linear Projection)
        y = torch.cat([x1, x2], dim=-1)

        y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))

        return self.norm_ff(y + y_ff), (attn1, attn2)


# class EfficientEncoderLayer(nn.Module):
#     """单路注意力+时序增强"""
#
#     def __init__(self, attention, d_model, d_ff=256, dropout=0.1, activation="relu"):
#         super().__init__()
#         self.attention = attention
#         # 轻量时序模块
#         self.temporal_aug = nn.Sequential(
#             nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
#             nn.GELU(),
#             nn.Dropout(dropout))
#
#         # 轻量FFN
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model))
#
#         self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # 注意力分支
#         attn_out, attn = self.attention(x, x, x, attn_mask)
#         x = self.norms[0](x + attn_out)
#
#         # 时序增强分支
#         temp_out = self.temporal_aug(x.transpose(1, 2)).transpose(1, 2)
#         x = self.norms[1](x + temp_out)
#
#         # FFN分支
#         ffn_out = self.ffn(x)
#         x = self.norms[2](x + ffn_out)
#         return x, attn


# class EfficientEncoderLayer(nn.Module):
#     """单路注意力+时序增强"""
#
#     def __init__(self, attention, d_model, d_ff=256, dropout=0.1, activation="relu"):
#         super().__init__()
#         self.attention = attention
#
#         self.temporal_aug = nn.Sequential(
#             nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
#             nn.GELU(),
#             nn.Dropout(dropout))
#
#
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model))
#
#         self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#
#         attn_out, attn = self.attention(x, x, x, attn_mask)
#         x = self.norms[0](x + attn_out)
#
#
#         temp_out = self.temporal_aug(x.transpose(1, 2)).transpose(1, 2)
#         x = self.norms[1](x + temp_out)
#
#
#         ffn_out = self.ffn(x)
#         x = self.norms[2](x + ffn_out)
#         return x, attn


class FeatureGroupTemporalEncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=256, dropout=0.1, nums_feature=0, activation="relu"):
        super().__init__()
        self.attention = attention

        self.temporal_aug = nn.Sequential(
            nn.Conv1d(nums_feature, nums_feature, 3, padding=1, groups=nums_feature),
            nn.GELU(),
            nn.Dropout(dropout))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        attn_out, attn = self.attention(x, x, x, attn_mask)
        x = self.norms[0](x + attn_out)


        temp_out = self.temporal_aug(x)
        x = self.norms[1](x + temp_out)


        ffn_out = self.ffn(x)
        x = self.norms[2](x + ffn_out)
        return x, attn