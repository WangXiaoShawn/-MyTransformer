import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        # 创建多个线性层，用于计算 Q、K、V
        #Newer versions of PyTorch allows nn.Linear to accept N-D input tensor, the only constraint is that the last dimension of the input tensor will equal in_features of the linear layer. 
        self.queries = nn.ModuleList([nn.Linear(embed_size, self.head_dim, bias=False) for _ in range(heads)])
        self.keys = nn.ModuleList([nn.Linear(embed_size, self.head_dim, bias=False) for _ in range(heads)])
        self.values = nn.ModuleList([nn.Linear(embed_size, self.head_dim, bias=False) for _ in range(heads)])
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)# 恢复到模型size 也就是词向量size 不然会有错误

    def forward(self, query, key, value, mask):
        # size query（N:batch_size,query_len,embing_size）
        N = query.shape[0] # batch size
        T = query.shape[1] # query length / key lenghth / value length -> sequence length
        queries = [linear(query) for linear in self.queries] # torch.Size([2, 10, 64]) -> list of 8 torch.Size([2, 10, 64])
        keys = [linear(key) for linear in self.keys]
        values = [linear(value) for linear in self.values]

        attention_heads = []
        #在多头注意力机制中，我们计算查询向量 (queries) 和键向量 (keys) 之间的相似度，得到 energy 矩阵。
        #随后，我们对 energy 矩阵应用 softmax，以便将相似度转换为权重（注意力分数）。
        for i in range(self.heads):
            energy = torch.einsum("nqd,nkd->nqk", [queries[i], keys[i]])#energy 矩阵表示查询向量 (queries) 和键向量 (keys) 之间的相似度
           
            if mask is not None:
                mask_expanded = mask.squeeze(1)
                # mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(N, self.heads, T, key.shape[1])
                energy = energy.masked_fill(mask_expanded == 0, float("-1e20")) #将 energy 中对应 mask 为 0 的位置的值替换为 float("-1e20")。
            attention = F.softmax(energy / (self.head_dim ** 0.5), dim=2)#nqk dim=2 表示在 T_k 维度上计算 softmax，即每个查询向量对于所有键的注意力权重。行
            out = torch.einsum("nqk,nvd->nqd", [attention, values[i]]) #吧value和attention相乘得到out
            attention_heads.append(out)
            
        concat_attention = torch.cat(attention_heads, dim=2)#将多头注意力机制的输出拼接起来 在d这个cacate
        out = self.fc_out(concat_attention)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))  # attention + query is the residual connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size # modelsize
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        """
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], device='cuda:0')
        """
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) # expand 方法将一维张量扩展为一个 N x seq_length 的二维张量
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask) # query, key, value都是out
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.attention = MultiHeadAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_expansion)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,value,key,src_mask,trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value,key,query,src_mask)
        return out
# 定义 Decoder 类

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=100,
        device="cuda"
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout=dropout,
            max_length=max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # 生成源序列的掩码
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # 生成目标序列的掩码，确保未来的词不会被注意到
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

def test_transformer():
    src_vocab_size = 10000
    trg_vocab_size = 10000
    src_pad_idx = 0
    trg_pad_idx = 0
    embed_size = 256
    num_layers = 6
    forward_expansion = 4
    heads = 8
    dropout = 0.1
    max_length = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size,
        num_layers,
        forward_expansion,
        heads,
        dropout,
        max_length,
        device
    ).to(device)

    N = 2  # batch size
    src_len = 15  # source sequence length
    trg_len = 10  # target sequence length

    src = torch.randint(0, src_vocab_size, (N, src_len)).to(device)
    trg = torch.randint(0, trg_vocab_size, (N, trg_len)).to(device)

    output = transformer(src, trg)
    assert output.shape == (N, trg_len, trg_vocab_size), "Output shape is incorrect"
    print("Forward pass test passed with output shape:", output.shape)


if __name__ == "__main__":
    test_transformer()