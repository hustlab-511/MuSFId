# from ResUnet import *
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pylab as plt

torch.manual_seed(10)

class Transformer_Encoder(nn.Module):
    def __init__(self, *, use_same_linear=False, input_data_dim, batches,
                 each_batch_dim, feed_forward_hidden_dim):
        super(Transformer_Encoder, self).__init__()

        # 必须保证头能被整除划分
        assert input_data_dim == batches * each_batch_dim

        self.use_same_linear = use_same_linear
        self.input_data_dim = input_data_dim

        self.batches = batches
        self.each_batch_dim = each_batch_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim

        self.d_k = each_batch_dim ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        # 是否采用同一个线性映射得到q、k、v
        self.linear_transfer = nn.Linear(self.input_data_dim,
                                         self.mid_data_dim) \
            if self.use_same_linear else nn.ModuleList([nn.Linear(self.each_batch_dim,
                                                                  self.each_batch_dim) for _ in range(99)])
        # 如果使用多头注意力将其降低为原来的通道
        self.combine_head_and_change_dim = nn.Linear(self.batches * self.each_batch_dim,
                                                     self.input_data_dim)


    def forward(self, same_output):

        same_output = same_output.repeat(1, 1, 3)
        output_data = torch.zeros((int(same_output.shape[0]), 1, self.each_batch_dim)).to(device)
        qq = same_output[:, :, 0:self.input_data_dim]
        kk = same_output[:, :, self.input_data_dim:self.input_data_dim * 2]
        vv = same_output[:, :, 2 * self.input_data_dim:3 * self.input_data_dim]
        for i in range(self.batches):
            q = qq[:,:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            k = kk[:,:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            v = vv[:,:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            q = self.linear_transfer[3*i+0](q) # 240 1 320
            k = self.linear_transfer[3*i+1](k) 
            v = self.linear_transfer[3*i+2](v)
            att = torch.matmul(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.d_k), v)

            output_data = torch.cat([output_data,
                                     att],
                                    dim=-1)

        output_data = output_data[:,:, self.each_batch_dim:]
        output_data = self.combine_head_and_change_dim(output_data)
        return output_data


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # 分多头  要求能够整除
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        # 分别对QKV做线性映射
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # 对输出结果做线性映射
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    # 输入QKV以及Mask  得到编码结果
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        # 把每个QKV中的单个分多头得到reshape后的QKV
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        # 对变形后的QKV做线性变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        # 计算得分
        attention = torch.softmax(energy / (self.embed_size ** (0.5)), dim=3).to(device)  # 240 5 10 10
        # 根据得分和value计算输出结果
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)
        # 对输出结果做线性映射
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # 自注意力编码 输入为Q K V Mask
        self.attention = SelfAttention(embed_size, heads)
        # 归一化
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # 前馈层
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    # 输入QKV Mask
    def forward(self, value, key, query):
        # 通过自注意力进行编码
        attention = self.attention(value, key, query)
        # 将  编码结果+Q   归一化
        x = self.dropout(self.norm1(attention + query))
        # 前馈层
        forward = self.feed_forward(x)
        # 再次进行归一化
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, embed_size, seq_length, elementlength=10):
        super(Encoder, self).__init__()
        self.seq_length = seq_length
        self.word_embedding = nn.Linear(elementlength, embed_size)
        self.position_embedding = nn.Embedding(seq_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size, heads=8, dropout=0, forward_expansion=4, )
            for _ in range(6)]
        )
        # self.linear = nn.Linear(embed_size, 10)

    def forward(self, input):
        N = input.shape[0]
        seq_length = self.seq_length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        p_embedding = self.position_embedding(positions)
        w_embedding = self.word_embedding(input)
        out = w_embedding + p_embedding
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        attention = self.attention(x, x, x)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query)
        return out


class Decoder(nn.Module):
    def __init__(self, embed_size, seq_length=30, elementlength=10):
        super(Decoder, self).__init__()
        self.device = device
        self.seq_length = seq_length
        self.word_embedding = nn.Linear(elementlength, embed_size)
        self.position_embedding = nn.Embedding(seq_length, embed_size)  # 位置编码
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size=embed_size, heads=8, forward_expansion=4, dropout=0, device=device)
             for _ in range(6)]
        )
        self.fc_out = nn.Linear(embed_size, elementlength)
        self.dropout = nn.Dropout(0.1)

    def forward(self, enc_out):
        N = enc_out.shape[0]
        seq_length = self.seq_length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        p_embedding = self.position_embedding(positions)
        w_embedding = enc_out
        x = (w_embedding + p_embedding)  # 向量 + 位置编码
        # DecoderBlock
        for layer in self.layers:
            x = layer(x, enc_out, enc_out)
        # 线性变换
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            seq_length=30,
            elementlength=0,
            src_pad_idx=0,
            trg_pad_idx=0,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            # device="cuda",
            max_length=1000
    ):
        super(Transformer, self).__init__()
        self.embed_size = 256
        self.encoder = Encoder(embed_size=self.embed_size, seq_length=seq_length, elementlength=elementlength)
        self.decoder = Decoder(embed_size=self.embed_size, seq_length=seq_length, elementlength=elementlength)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        # self.device = device
        self.linear = nn.Linear(self.embed_size, 100) # 需要与Unitedmodel的seqlength倍数一致
        # 降低对比学习输出特征维度
    def forward(self, src):  # shape  N 6 50
        # 编码
        enc_src = self.encoder(src)  # shape  N 6 256
        # 解码
        out = self.decoder(enc_src)  # shape  N 6 50
        features = self.linear(enc_src)
        return features, out


def train_TRM_net(*, model, data, origin, target, elementlength, lr, epoch):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    LossRecord = []
    for k in tqdm(range(epoch)):
        optimizer.zero_grad()
        masked = data.clone()

        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        masked = masked.view(data.shape[0], int(data.shape[2]/elementlength), elementlength).detach()
        origin = origin.view(data.shape[0], int(data.shape[2]/elementlength), elementlength).detach()
        features, ans = model(masked)
        loss1 = criterion(ans, origin)

        def Simloss(ans, target):
            U = ans * target  # 行和为正样本
            V = ans  # 行和为所有样本
            FU = torch.exp(U)
            FV = torch.exp(V)
            Usum = torch.sum(FU, dim=1)
            Vsum = torch.sum(FV, dim=1)
            output = -torch.log(Usum / Vsum)
            return torch.sum(output, dim=0)

        features = features.view(data.shape[0], 1, features.shape[1] * features.shape[2])
        features = F.normalize(features).squeeze(1)
        loss2 = criterion(torch.mm(features, features.t()), target) * 10

        loss = loss1 + loss2
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return model

class Classification(nn.Module):
    def __init__(self,input_data_dim,output_data_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_data_dim,10000,bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10000,output_data_dim,bias=False),
            # nn.LeakyReLU(inplace=True),
            nn.Softmax(dim=2)
        )
    def forward(self,output):
        output = self.linear(output)
        return output

def run_classification(*,TRMmodel,classify_model,data,classify_label,lr,epoch):
    optimizer = optim.Adam(classify_model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    LossRecord = []
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output = TRMmodel(data)
        output = classify_model(output)
        loss = criterion(output,classify_label.unsqueeze(0))
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return classify_model

