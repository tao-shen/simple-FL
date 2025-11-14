from .model_base import *
from .model_fn import *


# 定义优化器网络
class CoordinateLSTMOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(CoordinateLSTMOptimizer, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 定义LSTM层
        self.lstm1 = LSTMCell(input_size, hidden_size, bias=False)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        # 定义全连接层输出
        self.out = nn.Linear(hidden_size, output_size, bias=False)
        ### 如果要把输出直接作为梯度的话，bias会直接加上一个数量级差很多的值
        # 定义全连接层学习率
        self.lr = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, deltas, hx):
        # 先测试全局模型的优化
        avg = torch.mean(deltas, dim=-1)
        hidden, cell = self.lstm1(avg, hx)
        # hidden, cell = self.lstm1(hidden, (hidden, cell))
        # control=self.lr(hidden).squeeze(-1)*avg/self.args.lr_l
        # control=0
        # vec=self.out(cell).squeeze(-1)-control
        # control=torch.tanh(self.out(hidden)).squeeze(-1)
        # control=0
        # control=self.postprocess_control(control)
        # output=avg + control*self.args.lr_l
        vec=cell.mean(dim=-1)

        return vec, [hidden, cell]

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
    
    def postprocess_control(self, control, p=10):
        # 拆分处理后的输入
        gradient_norm, gradient_sign = control[:, 0], control[:, 1]
        
        # 近似恢复梯度的符号和未缩放前的值
        gradient_approx = gradient_sign / torch.exp(torch.tensor(p))
        # 从幅度裁剪中恢复
        gradient_magnitude_approx = torch.exp(gradient_norm * p)
        
        # 结合符号和幅度
        restored_gradient = gradient_magnitude_approx * torch.sign(gradient_approx)
        
        return restored_gradient*0.1



class LSTMCell(nn.Module):
    """
    自定义 LSTM 单元模块
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.b_i=0
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.b_f=0
        # 输出门参数
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.b_o=0
        # 单元状态参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.b_g=0
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化函数
        """
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for p in self.parameters():
            torch.nn.init.uniform_(p, -std, std)

    def forward(self, avg, state):
        inputs = self.preprocess_input(avg/0.01)
        
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
        # f_t=1
        c_t = (1-f_t**2*0.01) * c_prev + i_t * (g_t*0.01+avg.unsqueeze(dim=-1).expand_as(g_t))
        # 当前隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
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