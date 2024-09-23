import torch
import torch.nn as nn
from model import layers

class STGAT(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(STGAT, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STGATBlock(args.Kt, n_vertex, blocks[l][-1], blocks[l+1],  args.droprate, args.n_his, args.stblock_num))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x, distance, shortpath, quater, movement):
        conv_count = 0
        for idx, module in enumerate(self.st_blocks):
            try:
                x = module(x, distance, shortpath, quater, movement, conv_count)
                conv_count = conv_count + 1
            except Exception as e:
                print(f"Error at block {idx}: {e}")
                return 0

        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x