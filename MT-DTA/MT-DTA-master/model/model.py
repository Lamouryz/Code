import torch, math, itertools, torch.nn as nn, torch.functional as F, numpy as np
from torch.autograd import Variable
from arguments import argparser, logging
from datahelper import *

class Trans(nn.Module):
    def __init__(self, n_dims):
        super(Trans, self).__init__()
        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, in_channels = x.size()
        q = self.query(x).view(n_batch, in_channels, -1)
        k = self.key(x).view(n_batch, in_channels, -1)
        v = self.value(x).view(n_batch, in_channels, -1)
        kcont = torch.bmm(q.permute(0, 2, 1), k)
        poscont = torch.matmul(q.permute(0,2,1), k)
        vcont = torch.matmul(k.permute(0,2,1), v)
        energy = kcont + poscont
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        
        return out, kcont, poscont, vcont

class LayerNorm(nn.Module):
    def __init__(self, hsize, varE=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hsize))
        self.beta = nn.Parameter(torch.zeros(hsize))
        self.varE = varE

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.varE)
        return self.gamma * x + self.beta

class Muemb(nn.Module):
    def __init__(self, vasize, hsize, maxlen, dropout_rate):
        super(Muemb, self).__init__()
        self.wemb = nn.Embedding(vasize, hsize)
        self.posemb = nn.Embedding(maxlen, hsize)
        self.LayerNorm = LayerNorm(hsize)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputni):
        seq_length = inputni.size(1)
        posni = torch.arange(seq_length, dtype=torch.long, device=inputni.device)
        posni = posni.unsqueeze(0).expand_as(inputni)
        wemb = self.wemb(inputni)
        posemb = self.posemb(posni)
        emb = wemb + posemb
        emb = self.LayerNorm(emb)
        out = self.dropout(emb)
        return out

class mutcos(nn.Module):
    def __init__(self, xkcont, xposcont, xvcont, ykcont, yposcont, yvcont):
        super(mutcos, self).__init__()
        self.demb = Muemb(FLAGS.charsmiset_size, 128, FLAGS.max_smi_len, 0.003)
        self.xu = xkcont
        self.yu = xposcont
        self.vu = xvcont
        self.xd = ykcont
        self.yd = yposcont
        self.vd = yvcont
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, in_channels = x.size()
        mutu = torch.matmul(self.yu, (self.xd).transpose(1, 0))
        mutd = torch.matmul(self.xu, (self.yd).transpose(1, 0))
        attwu = torch.softmax(mutu, dim=1)
        attwd = torch.softmax(mutd, dim=1)
        energy = attwu + attwd
        attention = self.softmax(energy)
        out = torch.bmm(self.vd * self.vu, attention.permute(0, 2, 1))
        out = self.demb(x)
        contexemb = out.contiguous()
        contexembSh = contexemb.size()[:-2] + (self.all_head_size,)
        contexemb = contexemb.view(*contexembSh)
        out = contexemb.view(n_batch, in_channels)

class CasMol(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CasMol, self).__init__()
        self.FCn = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2, kernel_size=k_size, stride=1, padding=k_size//2)
        )
        self.SCn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2)
        )
        self.TCn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2)
        )
        self.TransMo = Trans(num_filters*3)
        self.out = nn.AdaptiveAvgPool1d(1)
        self.lineLayeru = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.lineLayerd = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.FCn(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.SCn(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.TCn(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.TransMo(x)
        output = self.out(x)
        output, kcont, poscont, vcont = self.TransMo(output)
        output = output.squeeze()
        outputu = self.lineLayeru(output)
        outputd = self.lineLayerd(output)
        output = self.reparametrize(outputu, outputd)
        return output, outputu, outputd, kcont, poscont, vcont


class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.lineLayerd = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, k_size):
        # print(x.shape)
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        # print(x.shape)
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.lineLayerd(x)
        return x


class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.prRe = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def forward(self, A, B):
        A = self.reg1(A)
        B = self.reg2(B)
        x = torch.cat((A, B), 1)
        x = self.prRe(x)
        return x

class net(nn.Module):
    def __init__(self, FLAGS , NUM_FILTERS , FILTER_LENGTH1 , FILTER_LENGTH2):
        super(net, self).__init__()
        self.embeddingU = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.embeddingD = nn.Embedding(FLAGS.charseqset_size, 128)
        self.Casnetu = CasMol(NUM_FILTERS, FILTER_LENGTH1)
        self.Casnetd = CasMol(NUM_FILTERS, FILTER_LENGTH2)
        self.mutcos = mutcos(NUM_FILTERS) 
        self.prRe = net_reg(NUM_FILTERS)
        self.decodeU = decoder(FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1,FLAGS.charsmiset_size)
        self.decodeD = decoder(FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2,FLAGS.charseqset_size)

    def forward(self, x, y, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        iniX = Variable(x.long()).cuda()
        x = self.embeddingU(iniX)
        x_embedding = x.permute(0, 2, 1)
        iniY = Variable(y.long()).cuda()
        y = self.embeddingD(iniY)
        y_embedding = y.permute(0, 2, 1)
        x1, pramX, pramvaX, xkcont, xposcont, xvcont = self.Casnetu(x_embedding)
        y1, pramY, pramvaY, ykcont, yposcont, yvcont = self.Casnetd(y_embedding)
        x, y = self.mutcos(xkcont, xposcont, xvcont, ykcont, yposcont, yvcont)
        out = self.prRe(x, y).squeeze()
        x1 = self.decodeU(x1, FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        y1 = self.decodeD(y1, FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        return out, x, y, iniX, iniY, pramX, pramvaX, pramY, pramvaY

# if __name__ == "__main__":
#
#     net()
