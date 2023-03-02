import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.functional as F
from arguments import argparser, logging
from datahelper import *
import math
import itertools

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

class Mutcros(nn.Module):

    def __init__(self, n_dims):
        super(Mutcros, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=n_dims*3, out_channels=n_dims*3, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=n_dims*3, out_channels=n_dims*3, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=n_dims*3, out_channels=n_dims*3, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out = nn.AdaptiveAvgPool1d(1)
        self.linearo = nn.Sequential(
            nn.Linear(n_dims*3, n_dims*3),
            nn.ReLU()
        )
        self.lineart = nn.Sequential(
            nn.Linear(n_dims*3, n_dims*3),
            nn.ReLU()
        )

    def repar(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def attval(self, cin):
        n_batch, in_channels, in_dim = cin.size()
        qcin = self.query_conv(cin).view(n_batch, in_channels, -1)
        kcin = self.query_conv(cin).view(n_batch, in_channels, -1)
        vcin = self.key_conv(cin).view(n_batch, in_channels, -1)

        return qcin, kcin, vcin

    def crosatt(self, co1, co2, co3, oglist, cin):
        n_batch, in_channels, in_dim = cin.size()
        crocc = torch.bmm(co1.permute(0, 2, 1), co2)
        crocp = torch.matmul(co1.permute(0, 2, 1), co2)
        croe = crocc + crocp
        croat = self.softmax(croe)
        extobatch = torch.bmm(co3, croat)
        extobatch = extobatch.view(n_batch, in_channels, -1)
        poolad = self.out(extobatch)
        sqout = torch.squeeze(poolad, 2)
        crosflo = self.linearo(sqout)
        crosflt = self.lineart(sqout)
        out = self.repar(crosflo + oglist[0], crosflt + oglist[1])

        return out

    def forward(self, x):
        xcin, ycin = x[0], x[7]
        xoglist = x[10:12]
        yoglist = x[12:14]
        xq, xk, xv = self.attval(xcin)
        yq, yk, yv = self.attval(ycin)
        xout = self.crosatt(xq, yk, xv, xoglist, xcin)
        yout = self.crosatt(yq, xk, yv, yoglist, ycin)

        return xout, yout


class Trans(nn.Module):
    def __init__(self, n_dims):
        super(Trans, self).__init__()

        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, in_channels, in_dim = x.size()
        q = self.query(x).view(n_batch, in_channels, -1)
        k = self.key(x).view(n_batch, in_channels, -1)
        v = self.value(x).view(n_batch, in_channels, -1)
        ck_cont = torch.bmm(q.permute(0, 2, 1), k)
        ck_poscont = torch.matmul(q.permute(0, 2, 1), k)
        att_ennergy = ck_cont + ck_poscont
        attention = self.softmax(att_ennergy)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, in_channels, -1)

        return out, q, k, v


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Muctfu(nn.Module):
    def __init__(self, num_fil, k_size):
        super(Muctfu, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_fil * 2, kernel_size=k_size, stride=1, padding=k_size // 2),

        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_fil, num_fil * 4, k_size, 1, k_size // 2),

        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_fil * 2, num_fil * 6, k_size, 1, k_size // 2),

        )
        self.cntrans = Trans(num_fil * 3)

        self.out = nn.AdaptiveAvgPool1d(1)
        self.linearo = nn.Sequential(
            nn.Linear(num_fil * 3, num_fil * 3),
            nn.ReLU()
        )
        self.lineart = nn.Sequential(
            nn.Linear(num_fil * 3, num_fil * 3),
            nn.ReLU()
        )

    def repar(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        xcin = out * torch.sigmoid(gate)
        x, q, k, v = self.cntrans(xcin)
        dgou = self.out(x)
        ousq = dgou.squeeze()
        outform = self.linearo(ousq)
        outforl = self.lineart(ousq)
        output = self.repar(outform, outforl)
        return output, outform, outforl, q, k, v, xcin


class decoder(nn.Module):
    def __init__(self, init_dim, num_fil, k_size, size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_fil * 3, num_fil * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_fil * 3, num_fil * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_fil * 2, num_fil, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_fil, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.lineart = nn.Linear(128, size)

    def forward(self, x, init_dim, num_fil, k_size):
        x = self.layer(x)
        x = x.view(-1, num_fil * 3, init_dim - 3 * (k_size - 1))
        x = self.convt(x)
        x = x.permute(0, 2, 1)
        x = self.lineart(x)
        return x


class net_reg(nn.Module):
    def __init__(self, num_fil):
        super(net_reg, self).__init__()
        self.prRe = nn.Sequential(
            nn.Linear(num_fil * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        self.prRe1 = nn.Sequential(
            nn.Linear(num_fil * 3, num_fil * 3),
            nn.ReLU()
        )

        self.prRe2 = nn.Sequential(
            nn.Linear(num_fil * 3, num_fil * 3),
            nn.ReLU()
        )

    def forward(self, A, B):
        A = self.prRe1(A)
        B = self.prRe2(B)
        x = torch.cat((A, B), 1)
        x = self.prRe(x)
        return x


class net(nn.Module):
    def __init__(self, FLAGS, num_fil, len_fild, len_filp):
        super(net, self).__init__()
        self.embedd = nn.Embedding(64, 128)
        self.embedp = nn.Embedding(25, 128)
        self.CNATDMo = Muctfu(num_fil, len_fild)
        self.CNATPMo = Muctfu(num_fil, len_filp)
        self.cros = Mutcros(num_fil)
        self.prRe = net_reg(num_fil)
        self.Decoderd = decoder(FLAGS.max_smi_len, num_fil, len_fild, 64)
        self.Decoderp = decoder(FLAGS.max_seq_len, num_fil, len_filp, 25)

    def forward(self, x, y, FLAGS, num_fil, len_fild, len_filp):
        init_X = Variable(x.long()).cuda()
        x = self.embedd(init_X)
        emb_X = x.permute(0, 2, 1)
        init_Y = Variable(y.long()).cuda()
        y = self.embedp(init_Y)
        emb_Y = y.permute(0, 2, 1)
        x, mu_x, logvar_x, qd, kd, vd, xcin = self.CNATDMo(emb_X)
        y, mu_y, logvar_y, qp, kp, vp, ycin = self.CNATPMo(emb_Y)
        val = [xcin, qd, qp, kd, kp, vd, vp, ycin, x, y, mu_x, logvar_x, mu_y, logvar_y]
        out_1, out_2 = self.cros(val)
        out = self.prRe(out_1, out_2).squeeze()
        x = self.Decoderd(x, FLAGS.max_smi_len, num_fil, len_fild)
        y = self.Decoderp(y, FLAGS.max_seq_len, num_fil, len_filp)
        return out, x, y, init_X, init_Y, mu_x, logvar_x, mu_y, logvar_y

