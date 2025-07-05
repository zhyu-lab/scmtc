import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mamba_ssm import Mamba
from einops import rearrange


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()


def rescale_distance_matrix(dist_matrix):
    constant_value = 1
    return (constant_value + math.exp(constant_value)) / (constant_value + torch.exp(constant_value - dist_matrix))


# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, adj_matrix, dist_matrix):
        d_k = K.size(2)
        if adj_matrix is not None:
            scores = torch.matmul(torch.matmul(Q, K.transpose(-1, -2)), adj_matrix) / np.sqrt(d_k)
        if dist_matrix is not None:
            scores = torch.matmul(nn.ReLU()(torch.matmul(Q, K.transpose(-1, -2))),
                                  rescale_distance_matrix(dist_matrix)) / np.sqrt(d_k)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, adj_matrix, dist_matrix):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, self.n_heads, -1).transpose(0, 1)
        K = self.W_K(input_K).view(batch_size, self.n_heads, -1).transpose(0, 1)
        V = self.W_V(input_V).view(batch_size, self.n_heads, -1).transpose(0, 1)

        context = ScaledDotProductAttention()(Q, K, V, adj_matrix, dist_matrix)
        context = context.transpose(0, 1).reshape(batch_size, -1)
        return context
        # return self.norm(residual + self.fc(context))


#  Position-wise Feed-Forward Networks
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class GlobalAndLocalAttentionLayer(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super(GlobalAndLocalAttentionLayer, self).__init__()
        self.d_model = d_model
        self.fc_down = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LeakyReLU()
        )
        self.fc_up = nn.Sequential(
            nn.Linear(d_model, input_dim),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.enc_self_attn_local = MultiHeadAttention(d_model // 2)
        self.enc_self_attn_global = MultiHeadAttention(d_model // 2)
        self.norm = nn.LayerNorm(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, x, adj_matrix, dist_matrix):
        x = self.fc_down(x)
        x1, x2 = torch.split(x, x.size(1) // 2, dim=-1)
        x_local = self.enc_self_attn_local(x1, x1, x1, adj_matrix, None)
        x_global = self.enc_self_attn_global(x2, x2, x2, None, dist_matrix)
        outputs = torch.cat((x_local, x_global), dim=-1)
        outputs = self.fc(outputs)
        outputs = self.norm(x + outputs)
        outputs = self.pos_ffn(outputs)
        outputs = self.fc_up(outputs)
        return outputs


class MambaBlock(nn.Module):
    def __init__(self, input_dim):
        """ Mamba Block
        """
        super(MambaBlock, self).__init__()
        self.d_model = 16

        self.mamba = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        res = x

        if x.size(1) % self.d_model != 0:
            padding = self.d_model - x.size(1) % self.d_model
            x = F.pad(x, (0, padding))

        x = rearrange(x, 'b (l d) -> b l d', d=self.d_model)
        x = self.mamba(x)
        x = rearrange(x, 'b l d -> b (l d)', d=self.d_model)

        if res.size(1) % self.d_model != 0:
            x, _ = torch.split(x, res.size(1), dim=-1)

        x = self.norm(x + res)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_dim):
        """ Conv Block
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(10, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.norm(x + self.conv(x))
        return x.squeeze()


# Decoder Mamba+Transformer+CNN
class EncoderLayer(nn.Module):
    def __init__(self, input_dim):
        super(EncoderLayer, self).__init__()

        self.att_layer = GlobalAndLocalAttentionLayer(input_dim)
        self.mamba_layer = MambaBlock(input_dim)
        self.conv_layer = ConvBlock(input_dim)

        self.fc = nn.Linear(3 * input_dim, input_dim)

    def forward(self, enc_inputs, adj_matrix, dist_matrix):
        res = enc_inputs

        x1 = self.att_layer(enc_inputs, adj_matrix, dist_matrix)
        x2 = self.mamba_layer(enc_inputs)
        x3 = self.conv_layer(enc_inputs)

        x = torch.cat((x1, x2, x3), dim=-1)
        x = self.fc(x)
        x = x + res

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, z_dim):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(input_dim) for _ in range(n_layers)])

        self.enc = nn.Linear(input_dim, z_dim)

    def forward(self, enc_inputs, adj_matrix, dist_matrix):
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, adj_matrix, dist_matrix)

        enc_outputs = self.enc(enc_outputs)
        return enc_outputs


# Decoder Mamba+CNN
class DecoderLayer(nn.Module):
    def __init__(self, input_dim):
        super(DecoderLayer, self).__init__()
        self.mamba_layer = MambaBlock(input_dim)
        self.conv_layer = ConvBlock(input_dim)
        self.fc = nn.Linear(2 * input_dim, input_dim)

    def forward(self, dec_inputs):
        res = dec_inputs
        x1 = self.mamba_layer(dec_inputs)
        x2 = self.conv_layer(dec_inputs)

        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
        x = x + res

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, n_layers, z_dim):
        super(Decoder, self).__init__()

        self.dec = nn.Linear(z_dim, input_dim)

        self.layers = nn.ModuleList([DecoderLayer(input_dim) for _ in range(n_layers)])

    def forward(self, enc_outputs):
        dec_outputs = self.dec(enc_outputs)
        for layer in self.layers:
            dec_outputs = layer(dec_outputs)

        return dec_outputs


class AutoEncoder(nn.Module):
    """
    This class implements an AutoEncoder with hybrid Mamba-Transformer-CNN architectures
    """

    def __init__(self, input_dim, n_layers=3, z_dim=64):
        super(AutoEncoder, self).__init__()
        if input_dim < 10000:
            self.conv_down = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_up = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
            seq_len = input_dim
        else:
            if input_dim >= 60000:
                kernel_size = 15
            else:
                kernel_size = 11
            self.conv_down = nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False)
            self.conv_up = nn.ConvTranspose1d(1, 1, kernel_size, stride=kernel_size, bias=True)
            seq_len = np.int32(input_dim / kernel_size)
        self.Encoder = Encoder(input_dim=seq_len, n_layers=n_layers, z_dim=z_dim)
        self.Decoder = Decoder(input_dim=seq_len, n_layers=n_layers, z_dim=z_dim)

        initialize_weights(self)

    def forward(self, enc_inputs, adj_matrix, dist_matrix):
        enc_inputs = self.conv_down(enc_inputs.unsqueeze(1))
        enc_inputs = enc_inputs.squeeze()
        enc_outputs = self.Encoder(enc_inputs, adj_matrix, dist_matrix)
        dec_outputs = self.Decoder(enc_outputs)
        dec_outputs = self.conv_up(dec_outputs.unsqueeze(1))
        dec_outputs = dec_outputs.squeeze()
        return dec_outputs, enc_outputs
