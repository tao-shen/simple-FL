import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from torchvision import datasets
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """
    使用狄利克雷分布将数据集非独立同分布地分配给多个客户端
    Args:
        train_labels: 训练数据的标签
        alpha: 狄利克雷分布的参数，越小表示分布越不均匀
        n_clients: 客户端数量
    """
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

def uniform_split_iid(train_labels, n_clients):
    """
    将数据集均匀地分配给多个客户端（IID情况）
    Args:
        train_labels: 训练数据的标签
        n_clients: 客户端数量
    """
    n_samples = len(train_labels)
    # 随机打乱数据
    indices = np.random.permutation(n_samples)
    # 计算每个客户端分配的样本数
    batch_size = n_samples // n_clients
    # 分配数据给每个客户端
    client_idcs = [indices[i * batch_size:(i + 1) * batch_size] for i in range(n_clients)]
    return client_idcs

def select_random_clients(client_idcs, n_select, seed):
    """
    从客户端列表中随机选择指定数量的客户端
    Args:
        client_idcs: 所有客户端的索引列表
        n_select: 要选择的客户端数量
        seed: 随机种子
    """
    np.random.seed(seed)
    selected_indices = np.random.choice(len(client_idcs), n_select, replace=False)
    return [client_idcs[i] for i in selected_indices]

def interpolate_heights(heights1, heights2, t):
    """
    在两个分布之间进行线性插值
    Args:
        heights1: 起始分布高度
        heights2: 目标分布高度
        t: 插值参数 (0-1)
    """
    return heights1 * (1 - t) + heights2 * t

def create_digit_images(train_data, train_labels):
    """
    为每个数字类别创建示例图像，并进行图像变换
    Returns:
        list of image arrays containing the digit images
    """
    sample_images = []
    for i in range(10):
        # 找到该数字的第一个样本
        idx = np.where(train_labels == i)[0][0]
        img = train_data.data[idx].numpy()
        # 左右翻转
        img = np.fliplr(img)
        # 逆时针旋转90度
        img = np.rot90(img)
        sample_images.append(img)
    return sample_images

def update(frame, ax, train_labels, client_idcs_list, iid_client_idcs, colors, num_cls, alpha, train_data, sample_axes):
    """
    更新动画帧
    Args:
        frame: 当前帧数
        ax: matplotlib子图对象列表
        train_labels: 训练数据标签
        client_idcs_list: 非IID客户端数据索引列表
        iid_client_idcs: IID客户端数据索引列表
        colors: 颜色列表
        num_cls: 类别数量
        alpha: alpha值列表
        train_data: 训练数据集
        sample_axes: 预先生成的数字图像数组列表
    """
    # 计算当前轮次和插值参数
    actual_round = frame // FRAMES_PER_ROUND
    next_round = min(actual_round + 1, N_ROUNDS - 1)
    t = (frame % FRAMES_PER_ROUND) / float(FRAMES_PER_ROUND)
    
    legend_artists = []
    
    # 首先处理IID情况
    ax[0].clear()
    
    # 设置固定的y轴范围和刻度
    max_height = 150
    tick_interval = 30
    
    ax[0].set_ylim(0, max_height)
    ax[0].yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
    
    # 为当前轮次和下一轮次随机选择不同的客户端
    selected_idcs_current = select_random_clients(iid_client_idcs, 10, actual_round)
    selected_idcs_next = select_random_clients(iid_client_idcs, 10, next_round)
    
    # 计算当前轮次和下一轮次的直方图高度
    hist_heights_current = [np.histogram(train_labels[idc], 
                          bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1))[0]
                          for idc in selected_idcs_current]
    hist_heights_next = [np.histogram(train_labels[idc], 
                       bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1))[0]
                       for idc in selected_idcs_next]
    
    # 在两个分布之间进行插值
    interpolated_heights = [interpolate_heights(curr, next_, t) 
                          for curr, next_ in zip(hist_heights_current, hist_heights_next)]
    
    total_heights = np.sum(interpolated_heights, axis=0)
    
    # 计算平滑曲线
    x = np.linspace(-0.5, num_cls-0.5, 200)
    y = np.zeros_like(x)
    for j in range(num_cls):
        y += total_heights[j] * np.exp(-(x - j)**2 / (2*SIGMA**2))
    
    actual_max = np.max(total_heights)
    if actual_max > 0:
        y = y * (actual_max / np.max(y)) * 1.1
    
    # 绘制堆叠柱状图
    for i, heights in enumerate(interpolated_heights):
        bar = ax[0].bar(range(num_cls), 
                   heights,
                   bottom=np.sum(interpolated_heights[:i], axis=0) if i > 0 else np.zeros(num_cls),
                   label=f"Client {chr(65+i)}",
                   color=colors[i],
                   alpha=BAR_ALPHA,
                   width=BAR_WIDTH,
                   zorder=1)
        if i == 0:
            legend_artists.append(bar)
    
    # 绘制平滑曲线和填充
    ax[0].fill_between(x, y, 0, alpha=CURVE_ALPHA, color='#2E5A88', zorder=2)
    ax[0].plot(x, y, '-', color='#2E5A88', linewidth=0.8, zorder=3, solid_capstyle='round')
    
    # 为每个标签添加边框
    for x_pos in range(num_cls):
        height = total_heights[x_pos]
        rect = plt.Rectangle((x_pos-0.25, 0), 0.5, height, 
                          fill=False, edgecolor='black', linewidth=0.5)
        ax[0].add_patch(rect)
    
    # 设置坐标轴
    ax[0].set_xticks(np.arange(num_cls))
    # 使用空字符串作为刻度标签，为图像留出空间
    ax[0].set_xticklabels([''] * num_cls)
    
    # 在x轴下方显示数字图像
    for i in range(num_cls):
        # 获取当前轴的变换
        trans = ax[0].get_xaxis_transform()
        # 在x轴刻度位置添加图像，去掉边框
        img_box = OffsetImage(sample_axes[i], zoom=DIGIT_ZOOM, cmap='gray')
        ab = AnnotationBbox(img_box, (i, DIGIT_OFFSET), xycoords=trans, box_alignment=(0.5, 1), pad=0)
        ax[0].add_artist(ab)
    
    # ax[0].set_xlabel('Sample Images', fontsize=10, labelpad=20)
    ax[0].tick_params(axis='both', which='major', labelsize=8)
    ax[0].grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax[0].set_axisbelow(True)
    
    # 添加边框
    for spine in ax[0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')
    
    # 设置标题
    ax[0].set_ylabel('Sample Count', fontsize=10, labelpad=10)
    ax[0].set_title('IID', fontsize=12, pad=10)
    
    # 处理非IID情况
    for col in range(3):
        g_current = []
        g_next = []
        np.random.seed(actual_round)  
        client_idcs_current = client_idcs_list[col].copy()
        np.random.shuffle(client_idcs_current)
        g_current.append(client_idcs_current[0:10])
        
        np.random.seed(next_round)
        client_idcs_next = client_idcs_list[col].copy()
        np.random.shuffle(client_idcs_next)
        g_next.append(client_idcs_next[0:10])
        
        # 计算当前轮次和下一轮次的直方图高度
        hist_heights_current = [np.histogram(train_labels[idc], 
                              bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1))[0]
                              for idc in g_current[0]]
        hist_heights_next = [np.histogram(train_labels[idc], 
                           bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1))[0]
                           for idc in g_next[0]]
        
        # 在两个分布之间进行插值
        interpolated_heights = [interpolate_heights(curr, next_, t) 
                              for curr, next_ in zip(hist_heights_current, hist_heights_next)]
        
        total_heights = np.sum(interpolated_heights, axis=0)
        
        # 清除上一帧的内容
        ax[col+1].clear()
        
        # 设置固定的y轴范围和刻度
        max_height = 150
        tick_interval = 30  # 每30一个刻度
        
        ax[col+1].set_ylim(0, max_height)
        ax[col+1].yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
        
        # 计算平滑曲线
        x = np.linspace(-0.5, num_cls-0.5, 200)
        y = np.zeros_like(x)
        for j in range(num_cls):
            y += total_heights[j] * np.exp(-(x - j)**2 / (2*SIGMA**2))
        
        # 确保曲线比柱状图略高
        actual_max = np.max(total_heights)
        if actual_max > 0:
            y = y * (actual_max / np.max(y)) * 1.1  # 增加10%的高度
        
        # 绘制堆叠柱状图
        artists = []
        for i, heights in enumerate(interpolated_heights):
            bar = ax[col+1].bar(range(num_cls), 
                       heights,
                       bottom=np.sum(interpolated_heights[:i], axis=0) if i > 0 else np.zeros(num_cls),
                       label=f"Client {chr(65+i)}",  # 使用大写字母A-J
                       color=colors[i],
                       alpha=BAR_ALPHA,
                       width=BAR_WIDTH,
                       zorder=1)
            artists.append(bar)
            
        if col == 0:
            legend_artists.extend(artists)
        
        # 绘制平滑曲线和填充
        ax[col+1].fill_between(x, y, 0, alpha=CURVE_ALPHA, color='#2E5A88', zorder=2)
        ax[col+1].plot(x, y, '-', color='#2E5A88', linewidth=0.8,
                     zorder=3, solid_capstyle='round')
        
        # 为每个标签添加边框
        for x_pos in range(num_cls):
            height = total_heights[x_pos]
            rect = plt.Rectangle((x_pos-0.25, 0), 0.5, height, 
                              fill=False, edgecolor='black', linewidth=0.5)
            ax[col+1].add_patch(rect)
        
        # 自定义子图样式
        ax[col+1].set_xticks(np.arange(num_cls))
        # 使用空字符串作为刻度标签，为图像留出空间
        ax[col+1].set_xticklabels([''] * num_cls)
        
        # 在x轴下方显示数字图像
        for i in range(num_cls):
            # 获取当前轴的变换
            trans = ax[col+1].get_xaxis_transform()
            # 在x轴刻度位置添加图像，去掉边框
            img_box = OffsetImage(sample_axes[i], zoom=DIGIT_ZOOM, cmap='gray')
            ab = AnnotationBbox(img_box, (i, DIGIT_OFFSET), xycoords=trans, box_alignment=(0.5, 1), pad=0)
            ax[col+1].add_artist(ab)
        
        # ax[col+1].set_xlabel('Sample Images', fontsize=10, labelpad=20)
        ax[col+1].tick_params(axis='both', which='major', labelsize=8)
        ax[col+1].grid(True, alpha=GRID_ALPHA, linestyle='--')
        ax[col+1].set_axisbelow(True)
        
        # 添加边框
        for spine in ax[col+1].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('black')
        
        # 设置标题和标签
        # if col == 0:
        #     ax[col+1].set_ylabel('Sample Count', fontsize=10, labelpad=10)  # 保持y轴标签
        ax[col+1].set_title(f'α = {alpha[col]}', fontsize=12, pad=10)
    
    # 设置总标题
    plt.suptitle(f'Data Distribution of 10/3400 Selected Clients at {actual_round+1}/{N_ROUNDS}-th Round', 
                fontsize=14, y=0.98, weight='bold')  # 加粗并调整标题
    
    return legend_artists

if __name__ == "__main__":
    # 可调整的参数说明：
    
    # 1. 数据分布相关参数
    N_CLIENTS = 3400  # 客户端总数，增加会使数据分布更细致，但计算时间更长
    N_ROUNDS = 3    # 动画轮次数，增加可以看到更多轮次的变化过程
    
    # alpha值含义：控制数据分布的不均匀程度
    # - alpha值越大（如10），数据分布越均匀，每个客户端的数据分布更接近整体分布
    # - alpha值越小（如0.01），数据分布越不均匀，每个客户端倾向于只包含少数类别的数据
    alpha = [1, 0.1, 0.01]  
    
    # 2. 动画相关参数
    FRAMES_PER_ROUND = 15  # 每轮的插值帧数，增加会使动画更平滑但更慢
    INTERVAL = 0        # 帧间隔时间(ms)，增加会使动画变慢
    FPS = 30            # 每秒帧数，减小会使动画变慢
    
    # 3. 可视化相关参数
    SIGMA = 0.4        # 曲线平滑度，增大会使曲线更平滑，减小会使曲线更接近原始直方图
    BAR_ALPHA = 0.7     # 柱状图透明度，增大更不透明，减小更透明
    BAR_WIDTH = 0.5     # 柱状图宽度，增大会使柱子更宽，减小会使柱子更窄
    CURVE_ALPHA = 0.09  # 曲线填充透明度，增大填充更明显，减小填充更淡
    GRID_ALPHA = 0.5    # 网格透明度，增大网格更明显，减小网格更淡
    DIGIT_ZOOM = 0.6    # 数字图片缩放比例，增大图片更大，减小图片更小
    DIGIT_OFFSET = -0.03  # 数字图片与x轴的距离，越接近0越靠近x轴
    
    # 4. 图像尺寸相关
    FIG_WIDTH = 20      # 图像宽度，增大图像更宽
    FIG_HEIGHT = 5      # 图像高度，增大图像更高
    DPI = 100          # 图像分辨率，增大图像更清晰但文件更大
    
    # 设置绘图样式
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['grid.alpha'] = GRID_ALPHA
    plt.rcParams['grid.color'] = '#cccccc'
    
    # 数据准备
    train_data = datasets.EMNIST(root=".", split="digits", download=True, train=True)
    test_data = datasets.EMNIST(root=".", split="digits", download=True, train=False)
    train_labels = np.array(train_data.targets)
    input_sz, num_cls = train_data.data[0].shape[0], len(train_data.classes)
    
    # 预先生成数字图像
    sample_axes = create_digit_images(train_data, train_labels)
    
    # 创建图形
    fig, ax = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.subplots_adjust(wspace=0.3, top=0.85)  
    
    # 生成IID数据分布
    iid_client_idcs = uniform_split_iid(train_labels, N_CLIENTS)
    
    # 为每个alpha值生成非IID客户端数据索引
    client_idcs_list = [
        dirichlet_split_noniid(train_labels, alpha=a, n_clients=N_CLIENTS)
        for a in alpha
    ]
    
    # 颜色列表
    cmap = 'Spectral'  
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, 10))  
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=N_ROUNDS * FRAMES_PER_ROUND,  
        fargs=(ax, train_labels, client_idcs_list, iid_client_idcs, colors, num_cls, alpha, train_data, sample_axes),
        interval=INTERVAL,  
        repeat=True
    )
    
    # 创建图例的虚拟矩形
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=f'Client {chr(65+i)}') 
                      for i in range(10)]
    
    # 添加图例
    fig.legend(handles=legend_elements,
              loc='center right', 
              bbox_to_anchor=(0.98, 0.5),
              title='Clients',
              fontsize=8,
              title_fontsize=10)
    
    # 保存动画
    anim.save('period_drift_animation.gif', writer='pillow', fps=FPS, dpi=DPI)
    plt.close()
