import torch.nn as nn
import torch
from torch.autograd import Variable
import math, copy
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, ft_size, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = 22
        self.time_len = time_len
        self.domain = domain

        if domain == "temporal" or domain == "mask_t":
            #temporal positial embedding
            pos_list = list(range(self.joint_num * self.time_len))
            #print("pos_list",pos_list)


        elif domain == "spatial" or domain == "mask_s":
            # spatial positial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)
            print(pos_list)
            #spatial:

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, ft_size)
        #position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, ft_size, 2).float() *
                             -(math.log(10000.0) / ft_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).cuda()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        #[batch, time, ft_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class MultiHeadedAttention(nn.Module):
    def __init__(self, h_num, h_dim, input_dim, dp_rate,domain):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        #assert d_model % h == 0
        # We assume d_v always equals d_k
        self.h_dim = h_dim # head dimension
        self.h_num = h_num #head num
        self.attn = None #calculate_att weight
        #self.att_ft_dropout = nn.Dropout(p=dp_rate)
        self.domain = domain  # spatial of  tempoal

        self.register_buffer('t_mask', self.get_domain_mask()[0])
        self.register_buffer('s_mask', self.get_domain_mask()[1])

        self.key_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )
        self.query_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )
        self.value_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.ReLU(),
                            nn.Dropout(dp_rate),
                                     )

    def get_domain_mask(self):
        # Sec 3.4
        time_len = 8
        joint_num = 22
        t_mask = torch.ones(time_len * joint_num, time_len * joint_num)
        filted_area = torch.zeros(joint_num, joint_num)

        for i in range(time_len):
            row_begin = i * joint_num
            column_begin = row_begin
            row_num = joint_num
            column_num = row_num
            t_mask[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= filted_area #Sec 3.4

        I = torch.eye(time_len * joint_num)
        s_mask = Variable((1 - t_mask)).cuda()
        #print('s_mask', s_mask)
        t_mask = Variable(t_mask + I).cuda()
        #print('t_mask', t_mask)
        return t_mask, s_mask

    def attention(self,query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        # [batch, time, ft_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        #print("scores",scores.shape)

        #print("self.t_mask.shape",self.t_mask.shape)
        #print("temporal first row ",self.t_mask[0])
        #print("spatial first row ",self.s_mask[0])
        #print("2nd row ",self.t_mask[1])
        #print("3nd row ",self.t_mask[2])
        #print("4nd row ",self.t_mask[3])
        #print(self.t_mask[2])
        #print("self.s_mask.shape",self.s_mask.shape)
        if self.domain is not None:
            #section 3.4 spatial temporal mask operation
            if self.domain == "temporal":
                scores *= self.t_mask  # set weight to 0 to block gradient
                #print("scores *= self.t_mask",scores[1][1][1])
                #print("scores.shape", scores.shape)
                scores += (1 - self.t_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax
                #print(" Temporal scores *= self.t_mask",scores[1][1][1])

            elif self.domain == "spatial":
                scores *= self.s_mask  # set weight to 0 to block gradient
                #print("spatial scores *= self.t_mask",scores[1][1][1])
                #print("spatial scores.shape", scores.shape)
                scores += (1 - self.s_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax
                #print(" Temporal scores *= self.t_mask",scores[1][1][1])


        # apply weight_mask to bolck information passage between ineer-joint
        #print("score.shape", self.domain, scores.shape)
        p_attn = F.softmax(scores, dim=-1)
        #print("p_attn: ",p_attn.shape)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x):
        "Implements Figure 2"
        nbatches = x.size(0) # [batch, t, dim]
        # 1) Do all the linear projections in batch from d_model => h x d_k
        #print('x input:', x.shape)

        query = self.query_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        #print('query:', query.shape)
        key = self.key_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        #print('key:', key.shape)
        value = self.value_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        #print('value:', key.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value) #[batch, h_num, T, h_dim ]
        #print("x, self.attn:",  x.shape, self.attn.shape)

            # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h_dim * self.h_num)#[batch, T, h_dim * h_num ]

        #print("after concatenation value",x.shape)
        return x


class Spatial_Temporal_Attention_Layer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, time_len, domain):
        #input_size : the dim of input
        #output_size: the dim of output
        #h_num: att head num
        #h_dim: dim of each att head
        #time_len: input frame number
        #domain: do att on spatial domain or temporal domain

        super(Spatial_Temporal_Attention_Layer, self).__init__()

        self.pe = PositionalEncoding(input_size, time_len, domain)
        #h_num, h_dim, input_dim, dp_rate,domain
        self.attn = MultiHeadedAttention(h_num, h_dim, input_size, dp_rate, domain) #do att on input dim

        self.ft_map = nn.Sequential(
                        nn.Linear(h_num * h_dim, output_size),
                        nn.ReLU(),
                        LayerNorm(output_size),
                        nn.Dropout(dp_rate),

                        )

        self.init_parameters()

    def forward(self, x):

        x = self.pe(x) #add PE
        x = self.attn(x) #pass attention model
        print("210 self.attn(x)",x.shape)
        x = self.ft_map(x)
        return x

    def init_parameters(self):
        model_list = [ self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)


#making model
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBranch_Attention_GG_DLM(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(MultiBranch_Attention_GG_DLM, self).__init__()

        h_dim = 32  # Dimensionality for attention mechanism (hidden dimension)
        h_num = 8  # Number of attention heads
        
        # Define the first fully connected layer (NN1)
        self.NN1 = nn.Sequential(
            nn.Linear(3, 128),  # Input size is 3 (x, y, z coordinates), output size is 128
            nn.ReLU(),  # Non-linear activation function
            LayerNorm(128),  # Normalize the output of the previous layer
            nn.Dropout(dp_rate),  # Dropout regularization to prevent overfitting
        )
        
        # Define the second fully connected layer (NN2)
        self.NN2 = nn.Sequential(
            nn.Linear(128, 256),  # Input size is 3, output size is 256
            nn.ReLU(),  # Non-linear activation function
            LayerNorm(256),  # Normalize the output
            nn.AvgPool2d((3, 2), stride=(1, 2))  # Average pooling to reduce the spatial size
        )
        
        # Spatial Attention Layer (to focus on spatial features)
        self.spatial_attention = Spatial_Temporal_Attention_Layer(
            input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len=8)
        
        # Temporal Attention Layer (to focus on temporal features)
        self.temporal_attention = Spatial_Temporal_Attention_Layer(
            input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len=8)
        
        # Final classification layer
        #self.cls = nn.Linear(128, num_classes)  # Elementwise addition/ element wise concatenation
        self.cls = nn.Linear(384, num_classes)  # Channel wise concatenation
    def forward(self, x):
        # Get the input shape: [batch_size, time_len, joint_num, 3]
        time_len = x.shape[1]  # Number of frames (e.g., 32 or 8)
        joint_num = x.shape[2]  # Number of keypoints (e.g., 21 or 22)
        
        # Reshape the input tensor to combine time_len and joint_num, keeping the 3 coordinate values
        # Reshaped to: [batch_size * time_len * joint_num, 3]
        original = x.reshape(-1, time_len * joint_num, 3)

        # Pass through NN1 to extract features from the input
        # Output: [batch_size * time_len * joint_num, 128]
        inmp = self.NN1(original)  

        # Branch-1: Spatial Attention -> Temporal Attention
        # Apply spatial attention to the output from NN1
        print("inmp.shape", inmp.shape)
        spatial_att_feature = self.spatial_attention(inmp)  # Output: [batch_size * time_len * joint_num, 256]
        
        # Apply temporal attention to the output of spatial attention
        temporal_att_feature = self.temporal_attention(spatial_att_feature)  # Output: [batch_size * time_len * joint_num, 256]
        
        # The final output of Branch 1 is the result of temporal attention
        Branch_1 = temporal_att_feature 
        
        # Branch-2: Temporal Attention -> Spatial Attention
        # Apply temporal attention to the output from NN1
        temporal_att_feature_2 = self.temporal_attention(inmp)  # Output: [batch_size * time_len * joint_num, 256]
        
        # Apply spatial attention to the output of temporal attention
        spatial_att_feature_2 = self.spatial_attention(temporal_att_feature_2)  # Output: [batch_size * time_len * joint_num, 256]
        
        # The final output of Branch 2 is the result of spatial attention
        Branch_2 = spatial_att_feature_2 
        
        # Branch-3: Pass through NN2 and apply average pooling
        res = self.NN2(inmp)  # Output: [batch_size * time_len * joint_num, 256]
        
        # Padding is applied to ensure dimensional consistency for concatenation
        Branch_3 = F.pad(res, pad=(0, 0, 2, 0), mode='constant', value=0)  # Padding applied to match the dimensions
        
        #element wise concatenation or addition
        #com = Branch_1+Branch_2+Branch_3  # We tested both case element wise concatenation Or channel wise concationa
        
        # Channel-wise Concatenation: Concatenate the outputs from all three branches along the channel dimension
        # This will stack the feature maps from Branch 1, Branch 2, and Branch 3
        print("Branch_1.shape:", Branch_1.shape)
        com = torch.cat((Branch_1, Branch_2, Branch_3), dim=-1)  # Concatenate along the channel (feature) dimension
        print("com.shape", com.shape)
        # Apply average pooling along the time dimension (sum across time frames and normalize by the number of time steps)
        x = com.sum(1) / com.shape[1]  # Average pooling over the time dimension (axis 1)

        # Pass the pooled features through the final classification layer
        pred = self.cls(x)  # Output: [batch_size, num_classes] (predictions for each class)

        return pred  # Return the class predictions
    
#call the model
input=torch.randn(2,8,22,3).cuda()
model=MultiBranch_Attention_GG_DLM(num_classes=5, dp_rate=0.1).cuda()
Output=model(input)