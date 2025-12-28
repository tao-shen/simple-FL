from .model_base import *
from .model_fn import *


# 定义聚合器和优化器网络
class Agg_Optimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, args):
        super(Agg_Optimizer, self).__init__()
        self.aggregator = CoordinateAggregator(input_size, hidden_size, args)
        self.optimizer = CoordinateOptimizer(
            input_size, hidden_size, output_size, num_layers, args
        )


# 定义聚合器网络
class CoordinateAggregator(nn.Module):
    def __init__(self, input_size, hidden_size, args):
        super(CoordinateAggregator, self).__init__()
        self.args = args
        self.input = nn.Linear(2, hidden_size)
        self.mh_attention = MultiheadAttention(
            hidden_size, 5, batch_first=True, bias=False
        )

    def forward(self, local_grads):
        x = self.preprocess_input(local_grads)
        x = self.input(x)
        x, _ = self.mh_attention(x, x, x)
        return x

    def preprocess_input(self, gradient, p=10):
        gradient_norm = torch.clamp(torch.log(torch.abs(gradient)) / p, min=-1)
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(p)) * gradient, min=-1, max=+1
        )
        return torch.stack([gradient_norm, gradient_sign], dim=-1)


# 定义优化器网络
class CoordinateOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, args):
        super(CoordinateOptimizer, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstms = nn.ModuleList(
            [
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=True)
                for i in range(num_layers)
            ]
        )
        self.lstms.append(LSTMCell_out(hidden_size, bias=True))

    def forward(self, x, hx):
        if hx is None:
            hx = [torch.zeros(2, self.hidden_size).to(x.device)] * self.num_layers

        for layer, lstm in enumerate(self.lstms):
            hx[layer] = lstm(x, hx[layer])
            x = hx[layer][0]

        return hx[-1][-1], hx


class CoordinateLSTMOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, args):
        super(CoordinateLSTMOptimizer, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.input = nn.Linear(2, hidden_size)
        self.mh_attention = MultiheadAttention(
            hidden_size, 1, batch_first=True, bias=False
        )
        # self.mh_attention=MultiHeadSelfAttention(hidden_size, 2)
        # 堆叠多个 LSTM 单元
        self.lstms = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=True)
            for i in range(num_layers)])
        self.lstms.append(LSTMCell_out(hidden_size, bias=True))
        # 模型的输出层也用lstm控制（其中c是模型参数，f是weight decay，i是学习率，f和i组成h，输入上一层lstm的h）

        # # 定义全连接层输出
        # self.out = nn.Linear(hidden_size, output_size, bias=True)
        # ### 如果要把输出直接作为梯度的话，bias会直接加上一个数量级差很多的值
        # # 定义全连接层学习率
        # self.lr = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, local_grads, hx):

        # assert hx[-1][-1] is None, 'No parameters input'
        if hx is None:
            hx = [torch.zeros(2, self.hidden_size).to(local_grads.device)]*self.num_layers

        # 不做聚合只avg后用lstm
        avg = torch.mean(local_grads, dim=-1) / self.args.lr_l
        x = self.preprocess_input(avg)
        # attention聚合
        # preprocessed = self.preprocess_input(local_grads)
        # x = self.input(preprocessed)
        # x = self.mh_attention(x,x,x)
        # x = x[0]
        # for i, lstm in enumerate(self.lstms):
        #     hx[i]=lstm(x, hx[i])
        #     x=hx[i][0]
        for layer, lstm in enumerate(self.lstms):
            hx[layer] = lstm(x, hx[layer])
            x = hx[layer][0]

        # hx[-1]包含了lr, weight_decay, model_param, hx[:-1]是前两层lstm的h和c
        return hx[-1][-1], hx

    def preprocess_input(self, gradient, p=10):
        # 定义输入预处理函数等效成两个裁剪
        # 裁剪幅值
        gradient_norm = torch.clamp(torch.log(torch.abs(gradient)) / p, min=-1)
        # 裁剪符号
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(p)) * gradient, min=-1, max=+1
        )

        processed_input = torch.stack([gradient_norm, gradient_sign], dim=-1)

        return processed_input


class LSTMCell_in(nn.Module):
    """
    自定义 LSTM 单元模块
    """
    def __init__(self, hidden_size, bias=True):
        super(LSTMCell_in, self).__init__()
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(1, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(3, 1))        
        self.b_i = nn.Parameter(torch.Tensor(1)) if bias else 0
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hf = nn.Parameter(torch.Tensor(3, 1))        
        self.b_f = nn.Parameter(torch.Tensor(1)) if bias else 0
        # 输出门参数
        # self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        # self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.b_o = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        # 单元状态参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hg = nn.Parameter(torch.Tensor(3, 1))        
        self.b_g = nn.Parameter(torch.Tensor(1)) if bias else 0
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化函数
        """
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for p in self.parameters():
            torch.nn.init.uniform_(p, -std, std)

    def forward(self, inputs, state):
        """
        前向传播函数
        """

        i_prev, f_prev, c_prev = state
        # 输入门
        _i=inputs @ self.W_ii + state.T @ self.W_hi + self.b_i
        i_t = torch.tanh(_i)*_i
        # 遗忘门
        _f=inputs @ self.W_if + state.T @ self.W_hf + self.b_f
        f_t = torch.tanh(_f)*_f
        # 单元状态
        _g=inputs @ self.W_ig + state.T @ self.W_hg + self.b_g
        g_t = torch.tanh(_g)

        i_t, f_t, g_t = i_t.squeeze(-1), f_t.squeeze(-1), g_t.squeeze(-1)
        # 当前单元状态
        c_t = (1-f_t) * c_prev + i_t * (g_t)

        return torch.stack([i_t, f_t, c_t], dim=0)


class LSTMCell_out(nn.Module):
    """
    自定义 LSTM 单元模块
    """
    def __init__(self, hidden_size, bias=True):
        super(LSTMCell_out, self).__init__()
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size,1))
        self.W_hi = nn.Parameter(torch.Tensor(3, 1))        
        self.b_i = nn.Parameter(torch.Tensor(1)) if bias else 0
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hf = nn.Parameter(torch.Tensor(3, 1))        
        self.b_f = nn.Parameter(torch.Tensor(1)) if bias else 0
        # 输出门参数
        # self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        # self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))        
        # self.b_o = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        # 单元状态参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hg = nn.Parameter(torch.Tensor(3, 1))        
        self.b_g = nn.Parameter(torch.Tensor(1)) if bias else 0
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化函数
        """
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for p in self.parameters():
            torch.nn.init.uniform_(p, -std, std)

    def forward(self, inputs, state):
        """
        前向传播函数
        """
            
        i_prev, f_prev, c_prev = state
        # 输入门
        _i=inputs @ self.W_ii + state.T @ self.W_hi + self.b_i
        i_t = torch.tanh(_i)*_i
        # 遗忘门
        _f=inputs @ self.W_if + state.T @ self.W_hf + self.b_f
        f_t = torch.tanh(_f)*_f
        # 单元状态
        _g=inputs @ self.W_ig + state.T @ self.W_hg + self.b_g
        g_t = torch.tanh(_g)
        
        i_t, f_t, g_t = i_t.squeeze(-1), f_t.squeeze(-1), g_t.squeeze(-1)
        # 当前单元状态
        c_t = (1-f_t) * c_prev + i_t * (g_t)

        return torch.stack([i_t, f_t, c_t], dim=0)


class LSTMCell(nn.Module):
    """
    自定义 LSTM 单元模块
    """
    def __init__(self, input_size, hidden_size, bias=False):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))        
        self.b_i = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))        
        self.b_f = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        # 输出门参数
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))        
        self.b_o = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        # 单元状态参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))        
        self.b_g = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化函数
        """
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for p in self.parameters():
            torch.nn.init.uniform_(p, -std, std)

    def forward(self, inputs, state):
        """
        前向传播函数
        """
        if state is None:
            h_t = torch.zeros(self.hidden_size, dtype=torch.float).to(inputs.device)
            c_t = torch.zeros(self.hidden_size, dtype=torch.float).to(inputs.device)
            state = (h_t, c_t)

        h_prev, c_prev = state
        # 输入门
        i_t = torch.sigmoid(inputs @ self.W_ii.T + h_prev @ self.W_hi.T + self.b_i)
        # 遗忘门
        f_t = torch.sigmoid(inputs @ self.W_if.T + h_prev @ self.W_hf.T + self.b_f)
        # 输出门
        o_t = torch.sigmoid(inputs @ self.W_io.T + h_prev @ self.W_ho.T + self.b_o)
        # 单元状态
        g_t = torch.tanh(inputs @ self.W_ig.T + h_prev @ self.W_hg.T + self.b_g)
        # 当前单元状态
        c_t = f_t * c_prev + i_t * g_t
        # 当前隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return torch.stack([h_t, c_t], dim=0)


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, batch_first=True, bias=False):
        super(MultiheadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear layers for query, key, value
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Final linear layer
        self.fc = nn.Linear(hidden_size, hidden_size)

        # Scaled factor
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Linear projections
        Q = self.query(x) # (batch_size, seq_length, hidden_size)
        K = self.key(x)   # (batch_size, seq_length, hidden_size)
        V = self.value(x) # (batch_size, seq_length, hidden_size)

        # Split into multiple heads and transpose
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V shape: (batch_size, num_heads, seq_length, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # attention_scores shape: (batch_size, num_heads, seq_length, seq_length)

        attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, seq_length, seq_length)

        output = torch.matmul(attention_weights, V)
        # output shape: (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        # output shape: (batch_size, seq_length, hidden_size)

        output = self.fc(output)
        # output shape: (batch_size, seq_length, hidden_size)

        return output, attention_weights


## 定义优化器网络
# class CoordinateLSTMOptimizer(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, args):
#         super(CoordinateLSTMOptimizer, self).__init__()
#         self.args = args
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#         # 定义LSTM层
#         self.lstm1 = LSTMCell(input_size, hidden_size)
#         self.lstm2 = nn.LSTMCell(input_size, hidden_size)
#         # 定义全连接层输出
#         self.out = nn.Linear(hidden_size, output_size)
#         ### 如果要把输出直接作为梯度的话，bias会直接加上一个数量级差很多的值
#         # 定义全连接层学习率
#         self.lr = nn.Linear(hidden_size, output_size)
#         # self.dropout=nn.Dropout(0.5)
#         self.lstms = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) if i == 0
#                                      else nn.LSTMCell(hidden_size, hidden_size)
#                                      for i in range(num_layers)])
#         self.activate=nn.Softmin(dim=-1)
#         self.m=0
#         self.v=0

#     def forward(self, local_grads, hx):
#         # Adam算法
#         # avg = torch.mean(local_grads, dim=-1)
#         # self.m=0.9*self.m+0.1*avg
#         # self.v=0.99*self.v+0.01*avg**2
#         # lr_ada=torch.sqrt(self.v) + 0.0001
#         # output=self.m/lr_ada
#         # lr_g=0.01
#         # return output, None, lr_g


#         # 先测试全局模型的优化
#         avg = torch.mean(local_grads, dim=-1)/self.args.lr_l
#         x = self.preprocess_input(avg)

#         # for i, lstm in enumerate(self.lstms):
#         #     hx[i]=lstm(x, hx[i])
#         #     x=hx[i][0]

#         hx[0] = self.lstm1(x, hx[0])
#         hx[1] = self.lstm2(x**2, hx[1])
#         hidden1 ,cell1 = hx[0]
#         hidden2 ,cell2 = hx[1]
#         control=torch.tanh(self.out(cell1)).squeeze(-1)
#         # control=0
#         # control=self.postprocess_control(control)
#         # control=self.dropout(control)###在control上加dropout没用，没影响
#         out=self.lr(cell2).squeeze(-1)
#         lr_ada=self.activate(out)*len(out)
#         output=(avg + control)*lr_ada
#         # output=(avg + control)*self.args.lr_l
#         lr_g=0.01

#         return output, hx, lr_g

#     def preprocess_input(self, gradient, p=10):
#         # 定义输入预处理函数等效成两个裁剪
#         # 裁剪幅值
#         gradient_norm = torch.clamp(torch.log(torch.abs(gradient)) / p, min=-1)
#         # 裁剪符号
#         gradient_sign = torch.clamp(
#             torch.exp(torch.tensor(p)) * gradient, min=-1, max=+1
#         )

#         processed_input = torch.stack([gradient_norm, gradient_sign], dim=-1)

#         return processed_input

#     def postprocess_control(self, control, p=10):
#         # 拆分处理后的输入
#         gradient_norm, gradient_sign = control[:, 0], control[:, 1]

#         # 近似恢复梯度的符号和未缩放前的值
#         gradient_approx = gradient_sign / torch.exp(torch.tensor(p))
#         # 从幅度裁剪中恢复
#         gradient_magnitude_approx = torch.exp(gradient_norm * p)

#         # 结合符号和幅度
#         restored_gradient = gradient_magnitude_approx * torch.sign(gradient_approx)

#         return restored_gradient*0.1

# def postprocess_control(self, control, p=10):
#     # 拆分处理后的输入
#     gradient_norm, gradient_sign = control[:, 0], control[:, 1]

#     # 近似恢复梯度的符号和未缩放前的值
#     gradient_approx = gradient_sign / torch.exp(torch.tensor(p))
#     # 从幅度裁剪中恢复
#     gradient_magnitude_approx = torch.exp(gradient_norm * p)

#     # 结合符号和幅度
#     restored_gradient = gradient_magnitude_approx * torch.sign(gradient_approx)

#     return restored_gradient*0.1
