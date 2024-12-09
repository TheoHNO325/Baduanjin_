import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset,DataLoader
from dataloader_2 import pth_Loader
from tqdm import tqdm

class MultiHeadSelfAttention(nn.Module):
    """
    model_dim: 模型维度，即输入和输出的向量维度。
    num_heads: 注意力头的数量。
    dropout_rate: Dropout率，防止模型过拟合，默认为0.1。
    """
    def __init__(self, model_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert model_dim % num_heads == 0, "model_dim 必须能整除注意力头的数量。"
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, inputs, attention_mask=None, target=None):
        """
        前向传播函数。
        参数：
        - inputs: 输入张量，形状为(batch_size, sequence_length, model_dim)。
        - mask: 掩码张量，形状为(batch_size, sequence_length, sequence_length)。
        返回：
        - output: 输出张量，形状为(batch_size, sequence_length, model_dim)。
        """
        batch_size, sequence_length, _ = inputs.shape
 
        # 对Query、Key和Value进行线性变换
        querys = self.query_projection(inputs)
        keys = self.key_projection(inputs)
        values = self.value_projection(inputs)
 
        # 进行矩阵分割以实现多头注意力
        querys = querys.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
 
        # 计算scaled dot-product attention,考虑注意力掩码
        attention_scores = torch.matmul(querys, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, sequence_length, -1)
            attention_scores = attention_scores.masked_fill(attention_mask == 1, float('-inf'))
        attention_probs = self.softmax(attention_scores)
        
        #应用训练阶段的dropout
        if target is not None:
            attention_probs = self.dropout(attention_probs) 
        attention_weights = torch.matmul(attention_probs, values).transpose(1, 2).reshape(batch_size, sequence_length, self.model_dim)
        output = self.output(attention_weights)
        return output, attention_probs

class FeedForward(nn.Module):
    def __init__(self, model_dim):
        super(FeedForward, self).__init__()
        self.model_dim = model_dim
        self.fc1 = nn.Linear(model_dim, 4*model_dim)
        self.fc2 = nn.Linear(4*model_dim, model_dim)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        output = self.fc2(self.relu(self.fc1(inputs)))
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        
        # 生成位置编码矩阵
        position = np.arange(max_len)[:, np.newaxis]  # shape (max_len, 1)
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))  # shape (dim/2,)
        self.encoding = torch.zeros(max_len, dim)
        self.encoding[:, 0::2] = torch.sin(torch.tensor(position) * torch.tensor(div_term))  # 偶数索引
        self.encoding[:, 1::2] = torch.cos(torch.tensor(position) * torch.tensor(div_term))  # 奇数索引
        self.encoding = self.encoding.unsqueeze(0).cuda()  # 增加batch维度，shape (1, max_len, dim)

    def forward(self, x):
        # x 的 shape 为 (batch_size, frame_num, dim)
        batch_size, frame_num, dim = x.size()

        # 确保位置编码仅与 frame_num 相匹配
        position_encoded = self.encoding[:, :frame_num, :].expand(batch_size, -1, -1)

        # 将位置编码添加到输入中
        x = x + position_encoded
        return x
    
def CalculateAttention(Q, K, V, mask):
    # 计算注意力分数
    attention = torch.matmul(Q, torch.transpose(K, -1, -2))
    
    # 确保 mask 的形状与 attention 的形状相匹配
    if mask is not None:
        # 对 mask 进行调整，以适应注意力分数的形状
        mask = mask.unsqueeze(1)  # 变为 (batch_size, 1, sequence_length)
        mask = mask.expand(-1, Q.size(1), -1, -1)  # 扩展到 (batch_size, num_heads, 1, sequence_length)
        attention = attention.masked_fill(mask == 0, -1e9)

    attention = torch.softmax(attention / math.sqrt(Q.size(-1)), dim=-1)
    attention = torch.matmul(attention, V)
    return attention

class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = math.sqrt(all_head_size)
    
    def forward(self,x,y,attention_mask):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        attention_mask = attention_mask.eq(0)

        attention = CalculateAttention(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output
    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_size, head_num, dropout_rate)
        self.cross_attn = Multi_CrossAttention(hidden_size, all_head_size, head_num)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.feedforward1 = FeedForward(hidden_size)
        self.feedforward2 = FeedForward(hidden_size)

    def forward(self, x, y, attention_mask = None):
        # self-attention
        x, _ = self.self_attn(x, attention_mask)
        x = self.feedforward1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        # cross-attention
        x = self.cross_attn(x, y, attention_mask)
        x = self.feedforward2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        return x
    
class Model(nn.Module):
    def __init__(self, input_size, frame_size, hidden_size, head_num, num_layers, dropout_rate=0.1):
        super().__init__()
        self.frame_size=frame_size
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.num_layers = num_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.embed_y = nn.Linear(input_size, hidden_size)
        self.pos_embed = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList([AttentionBlock(hidden_size, hidden_size, head_num, dropout_rate) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.linear2 = nn.Linear(hidden_size*frame_size, 256)
        self.linear3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, attention_mask=None):
        x_=x-y
        y_=torch.abs(x-y)
        x = self.embed(x_)
        y = self.embed_y(y_)
        x = self.pos_embed(x)
        y = self.pos_embed(y)
        
        for i,layer in enumerate(self.layers):
            x = layer(x, y, attention_mask)+x
        # print(""x.mean())
        x = self.norm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x
def evaluation(dataloader,model):
    model.eval()
    dataset_standard_path="/root/autodl-tmp/AlphaPose/dataset_processed14/standard_"
    t_pths=[]
    for i in range(14):
        t_pths.append(torch.load(dataset_standard_path+str(i)+'.pth'))
    for data in t_pths:
        data[np.isnan(data).bool()]=0
    attention_mask = torch.zeros(1,200,200).cuda()
    tot=0
    acc=0
    score_ev=0
    for st,ref,sc in tqdm(dataloader):
        x=st.cuda()
        if sc.item() < 0.01:
            continue
        gt_id=-1
        scores=[]
        for i in range(14):
            y = t_pths[i].cuda().unsqueeze(0)
            if np.abs(y.mean().item() - ref.mean().item())<1e-4:
                gt_id=i
            out=model(x,y,attention_mask).item()
            scores.append(out)
        scores=np.array(scores)
        tot+=1
        if gt_id == np.argmax(scores):
            acc+=1
            score_ev+=1-np.abs(scores[gt_id]-sc.item())
    print("accuracy",acc/tot,"score_ev",score_ev/tot)
def train_one_epoch(dataloader,model,opt,loss_fn,input_size, frame_size,batch_size=32):
    total_loss = []
    model.train()
    attention_mask = torch.zeros(batch_size, frame_size, frame_size).bool().cuda()
    # attention_mask[:, :, 5:] = True
    i = 0
    for i,(x,y,score) in enumerate(dataloader):
        if len(score) != batch_size: continue
        x,y = x.cuda(),y.cuda()
        # print(x.shape,y.shape)
        score = score.unsqueeze(1).float().cuda()
        output = model(x, y, attention_mask)
        # print(output,x.mean(),y.mean())
        opt.zero_grad()
        loss = loss_fn(score,output)
        loss.backward()
        opt.step()
        # print(f"times{i},loss={loss}")
        total_loss.append(loss.item())
        i += 1

    return np.array(total_loss)
def roll(x,delta):
    x_shape=x.shape
    video_length=x_shape[1]
    bs=x_shape[0]
    all_idx = [x for x in range(video_length)]
    return x.clone()[:,all_idx[delta:]+all_idx[:delta],:]
if  __name__ == '__main__':

    dataset_path = "/root/autodl-tmp/AlphaPose/dataset_processed14/overall.pth"
    pth = torch.load(dataset_path)
    pth_train=[]
    pth_test=[]
    for i in range(len(pth)):
        if i % 9 != 0:
            pth_train.append(pth[i])
        else:
            pth_test.append(pth[i])
    pth = pth_Loader(pth_train)
    pth_test = pth_Loader(pth_test)
    
    pthloader = DataLoader(pth,batch_size=32,shuffle=True)
    testloader = DataLoader(pth_test,batch_size=1,shuffle=False)

    model = Model(input_size=42, frame_size=200, hidden_size=64, head_num=8, num_layers=12).cuda()
    model.load_state_dict(torch.load('./model.pth'))
    evaluation(testloader,model)