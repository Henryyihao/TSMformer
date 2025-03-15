import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(RelativePositionEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        # Learnable relative position embeddings
        self.relative_position_matrix = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model)  # Covers relative positions [-max_len, max_len]
        )

    def forward(self, seq_len):
        """
        Args:
            seq_len: Sequence length (T)
        Returns:
            r: Relative position encoding [T, T, d_model]
        """
        range_vec = torch.arange(seq_len, device=self.relative_position_matrix.device)
        relative_positions = range_vec.view(-1, 1) - range_vec.view(1, -1)
        relative_positions = relative_positions + self.max_len - 1  # Shift to [0, 2*max_len-2]
        r = self.relative_position_matrix[relative_positions]  # Shape: [T, T, d_model]
        return r

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h', c_in=24):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq, c_in=c_in)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# relevant pos
# class DataEmbedding_inverted_O(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='conv', freq='h', dropout=0.1):
#         super(DataEmbedding_inverted_O, self).__init__()
#
#         self.value_embedding = nn.Linear(c_in, d_model)
#         self.relative_position_embedding = RelativePositionEncoding(500, d_model)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x, x_mark=None):
#         """
#         Args:
#             x: Input data [Batch, Time, 1] (e.g., target variable values)
#             x_mark: Covariates [Batch, Time, num_covariates]
#         Returns:
#             x: Embedded data with relative position encoding [Batch, Variate, d_model]
#         """
#         B, T, C = x.shape  # [Batch, Time, 1]
#         x = x.permute(0, 2, 1)  # [Batch, 1, Time]
#
#         # Concatenate covariates if provided
#         if x_mark is not None:
#             x_mark = x_mark.permute(0, 2, 1)  # [Batch, num_covariates, Time]
#             x = torch.cat([x, x_mark], dim=1)  # [Batch, Variate, Time]
#
#         # Apply value embedding
#         x = self.value_embedding(x)  # [Batch, Variate, Time] -> [Batch, Variate, d_model]
#
#         # Add relative position encoding
#         r = self.relative_position_embedding(T)  # Shape: [T, T, d_model]
#         r = r.permute(2, 0, 1)  # [T, T, d_model] -> [d_model, T, T]
#
#         # Fuse embeddings with relative positions
#         x = torch.einsum('ijk,kmn->ijk', x, r)  # Combine embeddings [B, V, T] with relative positions
#
#         return self.dropout(x)  # Output shape: [Batch, Variate, d_model]

# position
class DataEmbedding_inverted_O(nn.Module):
    def __init__(self, c_in, d_model, embed_type='conv', freq='h', dropout=0.1):
        super(DataEmbedding_inverted_O, self).__init__()

        # 用线性层对输入特征嵌入
        self.value_embedding = nn.Linear(c_in, d_model)

        # 创建可学习的位置嵌入
        self.position_embedding = nn.Parameter(torch.randn(1, c_in * 10, d_model))
        self.position_embedding.data = F.normalize(self.position_embedding.data, dim=-1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # 交换维度，将 [Batch, Time, Variate] -> [Batch, Variate, Time]
        x = x.permute(0, 2, 1)

        # 若有额外特征，拼接
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)

        # 线性映射到 d_model 维度
        x = self.value_embedding(x) + self.position_embedding[:, :x.size(1), :]

        # 加入位置嵌入
        # x = x + self.position_embedding[:, :x.size(1), :]

        return self.dropout(x)



# multi head
# class DataEmbedding_inverted(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='conv', freq='h', dropout=0.1):
#         super(DataEmbedding_inverted, self).__init__()
#
#         # 创建多个线性层，每个头负责一部分特征
#         self.head_dim = d_model // 4
#         self.heads = nn.ModuleList([
#             nn.Linear(c_in, self.head_dim) for _ in range(4)
#         ])
#
#         # 最后用线性层合并每个头的输出
#         self.final_linear = nn.Linear(self.head_dim * 4, d_model)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x, x_mark=None):
#         # 交换维度，将 [Batch, Time, Variate] -> [Batch, Variate, Time]
#         x = x.permute(0, 2, 1)
#
#         # 若有额外特征（如时间戳），在特征维度上拼接
#         if x_mark is not None:
#             x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)
#
#         # 对每个头分别嵌入，然后拼接
#         x = torch.cat([head(x) for head in self.heads], dim=-1)
#
#         # 交换维度回到 [Batch, Time, d_model]，并通过最后线性层
#         x = self.final_linear(x)
#
#         return self.dropout(x)


# conv1d
# class DataEmbedding_inverted(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='conv', freq='h', dropout=0.1):
#         super(DataEmbedding_inverted, self).__init__()
#
#         # 使用卷积嵌入，将输入特征映射到 d_model 维度
#         self.value_embedding = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1)
#         self.dropout = nn.Dropout(p=dropout)
#
    # def forward(self, x, x_mark=None):
    #     # 交换维度，将 [Batch, Time, Variate] -> [Batch, Variate, Time]
    #     x = x.permute(0, 2, 1)
    #
    #     # 若有额外特征（如时间戳），在特征维度上拼接
    #     if x_mark is not None:
    #         x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)
    #
    #     # 卷积嵌入
    #     x = x.permute(0, 2, 1)
    #
    #     x = self.value_embedding(x)
    #
    #     # 交换维度回到 [Batch, Time, d_model]
    #     x = x.permute(0, 2, 1)
    #
    #     return self.dropout(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        # 使用适当的填充确保因果性
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=0, dilation=dilation)

    def forward(self, x):
        # 在序列的前面填充，确保因果性
        x = nn.functional.pad(x, (self.padding, 0))  # 只填充左侧
        out = self.conv(x)
        return out


class VariableWeighting(nn.Module):
    def __init__(self, enc_in, d_model):
        super(VariableWeighting, self).__init__()
        self.variable_weights = nn.Parameter(torch.randn(enc_in, d_model))

    def forward(self, x):
        # x: [Batch, Variate, d_model]
        return x * self.variable_weights


class DataEmbedding_inverted_conv(nn.Module):
    def __init__(self, c_in, d_model, nums_feature, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted_conv, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.causal_conv = CausalConv1d(d_model, d_model, kernel_size=3, dilation=2)
        self.var_weighting = VariableWeighting(nums_feature+d_inp, d_model)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.causal_conv(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.var_weighting(x)  # [Batch, Variate, d_model]
        x = self.dropout(x)
        # x: [Batch Variate d_model]
        return x

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# 原始
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
