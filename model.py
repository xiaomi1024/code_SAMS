import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda')

class Att(nn.Module):
    def __init__(self, hidden_size, attention_size, num_layers):
        super(Att, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_layers = num_layers

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * num_layers, attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size))
        nn.init.constant_(self.w_omega, 0.001)
        nn.init.constant_(self.u_omega, 0.001)
    def forward(self, x, gru_output):
        mask = torch.sign(torch.abs(torch.sum(x, axis=-1))).to(device)
        attn_tanh = torch.tanh(torch.matmul(gru_output, self.w_omega))
        attn_hidden_layer = torch.matmul(attn_tanh, self.u_omega)
        paddings = torch.ones_like(mask) * (-10e8)
        attn_hidden_layer = torch.where(torch.eq(mask, 0), paddings, attn_hidden_layer)
        alphas = F.softmax(attn_hidden_layer, 1)
        attn_output = torch.sum(gru_output * torch.unsqueeze(alphas, -1), 1)
        return attn_output

class SAMS(nn.Module):
    def __init__(self, num_classes, in1, in2):
        super(SAMS, self).__init__()
        #####
        self.hidden_size1 = 128
        self.hidden_size2 = 128
        self.attention_size = 4
        self.num_layers = 2
        self.direction = 2
        self.gru1 = nn.GRU(in1, self.hidden_size1, self.num_layers, batch_first=True, bidirectional=True)
        self.gru11 = nn.GRU(in1, self.hidden_size1, self.num_layers, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(in2, self.hidden_size2, self.num_layers, batch_first=True, bidirectional=True)
        self.gru22 = nn.GRU(in2, self.hidden_size2, self.num_layers, batch_first=True, bidirectional=True)

        self.Att1 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Att11 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Atts1 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Att2 = Att(self.hidden_size2, self.attention_size, self.num_layers)
        self.Att22 = Att(self.hidden_size2, self.attention_size, self.num_layers)
        self.Attt2 = Att(self.hidden_size2, self.attention_size, self.num_layers)
        self.Atts11 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Attt22 = Att(self.hidden_size2, self.attention_size, self.num_layers)



        self.st_all_c = nn.Sequential(
            nn.Linear((self.direction * self.hidden_size1 + self.direction * self.hidden_size2) * 4, 64),
            nn.ReLU())

        self.st = nn.Sequential(
            nn.Linear((self.direction * self.hidden_size1 + self.direction * self.hidden_size2) *2, 64),
            nn.ReLU())
        self.s = nn.Sequential(
            nn.Linear(self.direction * self.hidden_size1, 64),
            nn.ReLU())
        self.t = nn.Sequential(
            nn.Linear(self.direction * self.hidden_size2, 64),
            nn.ReLU())
        self.ss = nn.Sequential(
            nn.Linear(self.direction * self.hidden_size1, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))
        self.tt = nn.Sequential(
            nn.Linear(self.direction * self.hidden_size2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.c_fc = nn.Linear(64, num_classes)

    def mul(self, a, b):
        attn = torch.bmm(a, b.permute(0, 2, 1))
        output = torch.bmm(F.softmax(attn, 2), b)
        return output
    def mut(self, q, k):
        attn = torch.bmm(q.unsqueeze(2), k.unsqueeze(1))
        output = torch.bmm(F.softmax(attn, 2),  k.unsqueeze(2)).view(q.shape[0],-1) + q
        return output
    def forward(self, x, xf, training, stf):
        x = x[ : , :, 0 : 78]
        if stf == 'st':
            # # #################### GRU ##############################################
            h1 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size1).to(device)  # 2 for bidirection
            h2 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size2).to(device)
            h11 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size1).to(device)  # 2 for bidirection
            h22 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size2).to(device)
            #################feature learning########################
            out1, _ = self.gru1(x, h1)
            out2, _ = self.gru2(xf, h2)
            out11, _ = self.gru11(x, h11)
            out22, _ = self.gru22(xf, h22)

            ################IFM1#################
            out_s = self.mul(out1, out11) + out1
            out_t = self.mul(out2, out22) + out2

            out_ss = self.mul(out11, out1) + out11
            out_tt = self.mul(out22, out2) + out22


            outa1 = self.Att1(x, out1)
            outa2 = self.Att2(xf, out2)
            outa11 = self.Att11(x, out11)
            outa22 = self.Att22(xf, out22)

            out_s1 = self.Atts1(x, out_s)
            out_t1 = self.Attt2(xf, out_t)
            out_s2 = self.Atts11(x, out_ss)
            out_t2 = self.Attt22(xf, out_tt)

            ################IFM2#################
            a = self.mut(outa11, outa1)
            b = self.mut(outa22, outa2)

            aa = self.mut(outa1, outa11)
            bb = self.mut(outa2, outa22)


            c1 = torch.cat((a, b), 1)
            c11 = torch.cat((aa, bb), 1)

            c2 = torch.cat((out_s1, out_t1), 1)
            c22 = torch.cat((out_s2, out_t2), 1)


            # #######ALL center #####
            c_all_c = torch.cat((torch.cat((c1, c2), 1), torch.cat((c11, c22), 1)), 1)
            out = self.st_all_c(c_all_c)

            out = self.c_fc(out)
            if training == 1:
                return out, outa1, outa2, outa11, outa22, self.ss(outa1), self.tt(outa2)
            else:
                return out
# demo
# model = SAMS(4, 78, 768).to(device)
# input1 = torch.randn(1, 500, 78).to(device)
# input2 = torch.randn(1, 150, 768).to(device)
# out = model(input1, input2, 0, 'st')
# print(out)

