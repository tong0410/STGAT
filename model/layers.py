import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x

        return x

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), torch.sigmoid(x_q))

        return x

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, n_his, stblock_num, dropout=0.2, alpha=0.2, num_heads=8,  ratio = 0.5):
        super(MultiHeadGraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_heads = num_heads
        self.ratio = ratio
        self.d_k = out_features // num_heads
        self.movement_linear = nn.Linear(3, 1, bias=True)
        self.quater_linear = nn.Linear(4, 1, bias=True)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.conv_layer = nn.ModuleList()
        for i in range(stblock_num):
            out_channels = n_his - (i + 1) * 4 + 2
            self.conv_layer.append(nn.Conv2d(in_channels=n_his, out_channels=out_channels, kernel_size=1))



    def forward(self, input, distance, shortpath, quater, movement, conv_count):
        batch_size, C_in, T, M = input.size()

        input = input.permute(0, 2, 3, 1).contiguous().view(batch_size * T, M, C_in)
        distance = self.conv_layer[conv_count](distance)
        shortpath = self.conv_layer[conv_count](shortpath)
        movement= self.movement_linear(movement) # Convert to scalars
        movement = movement.squeeze(-1)
        movement = self.conv_layer[conv_count](movement)

        quater= self.quater_linear(quater) # Convert to scalars
        quater = quater.squeeze(-1)
        quater = self.conv_layer[conv_count](quater)

        h = torch.matmul(input, self.W)

        h_heads = h.view(batch_size * T, M, self.num_heads, self.d_k).transpose(2, 1)

        attention_outputs = []
        for i in range(self.num_heads):
            h_head = h_heads[:, i, :, :]
            h_flat = h_head.view(batch_size * T, M, -1)

            h_repeated1 = h_flat.unsqueeze(2).repeat(1, 1, M, 1)
            h_repeated2 = h_flat.unsqueeze(1).repeat(1, M, 1, 1)
            h_concat = torch.cat([h_repeated1, h_repeated2], dim=3)

            a_input = h_concat.view(batch_size * T * M * M, -1)
            # e = torch.matmul(a_input, self.a).view(batch_size, T, M, M)
            e = torch.matmul(a_input, self.a).view(batch_size, T, M, M)

            e = e + movement + quater

            if distance is not None:
                e = e * distance.to(e.device).float()
                e = e.masked_fill(e == 0, float("-inf"))
            if shortpath is not None:
                e = self.ratio * e + (1 - self.ratio) * (shortpath.to(e.device).float())
                e = e.masked_fill(e == 0, float("-inf"))

            attention = F.softmax(e, dim=-1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            attention_reshaped = attention.view(batch_size * T, M, M)
            h_prime = torch.matmul(attention_reshaped, h_flat)
            # h_prime = h_prime.view(batch_size, T, self.d_k)

            attention_outputs.append(h_prime)

        h_prime_concat = torch.cat(attention_outputs, dim=-1)
        output = h_prime_concat.view(batch_size, T, M, -1)

        return output


class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, n_his, stblock_num):
        super(GATLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.gat = MultiHeadGraphAttention(c_out, c_out, n_his, stblock_num)

    def forward(self, x,  distance, shortpath, quater, movement, conv_count):
        # residual connection
        x_at_re = self.align(x)
        x_at = self.gat(x_at_re,  distance, shortpath, quater, movement, conv_count)
        x_at = x_at.permute(0, 3, 1, 2)
        x_total = torch.add(x_at, x_at_re)
        return x_total

class STGATBlock(nn.Module):
    def __init__(self, Kt, n_vertex, last_block_channel, channels, droprate, n_his, stblock_num):
        super(STGATBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex)
        self.gat = GATLayer(channels[0], channels[1], n_his, stblock_num)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x,  distance, shortpath, quater, movement, conv_count):
        x = self.tmp_conv1(x)
        x = self.gat(x , distance, shortpath, quater, movement, conv_count)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x

class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x

