import numpy as np
import concurrent.futures
from scipy.signal import windows, welch, csd
from scipy.io import loadmat
import time

# 文件加载
mat = loadmat("homework/White_IO.mat")
# 计算H传递函数, 参数：
fs = 48000      # 采样率
nfft = 48000 * 3        # 信号帧
win = windows.hann(nfft)    # 汉宁窗
random_in = mat['random_in']       # 输入    8 * 44100(8个音响, 44100采样)
len_sig = len(random_in[0])      # 信号长度
response = mat['Response']          # 输出    8 * 8 * 44100(8个音响, 8个麦克风, 44100采样)
L = 8                               # 扬声器
M = 8                               # 麦克风


# 传递函数计算
def transport():
    H = []
    # H = Gxy / Gxx
    for i in range(L):
        # 返回功率谱 f 采样频率数组 Pxx x的功率谱密度 (8 * n_fft/2+1)
        Gxx = welch(random_in[i], fs, win, nperseg=nfft, axis=-1)[1]
        # 返回功率谱 f 采样频率数组 Pxy x,y的交叉功率谱密度    (8 * n_fft/2+1)
        Gxy = csd(random_in[i], response[i], fs, win, nperseg=nfft, axis=-1)[1]
        # 计算 H
        temp = Gxy / Gxx
        H.append(temp)
    # H: L(8) * M(8) * (n_fft/2+1)
    # H_list: (n_fft/2+1) * M(8) * L(8) 数据: 每个频点(8*8) 每个麦克风 -> 8个扬声器输入
    H_list = list(np.array(H).transpose([2, 1, 0]))
    return H_list


# 计算逆矩阵函数
def inversion(H_list, epsilon):
    C_list = []
    H = np.array(H_list)
    # 矩阵H_list 48000*3/2+1 M L
    # 吉洪诺夫正则化方法所要返回所需加入单位矩阵
    gamma = epsilon * np.eye(M)
    for i in range(H.shape[0]):
        H_f = H[i]     # 8 * 8
        C_f = (np.linalg.pinv(H_f.T.conjugate() @ H_f) + gamma) @ H_f.T.conjugate()   # 8 * 8
        C_list.append(C_f)
    return C_list


# 多进程
def inversion1(H_list, epsilon):
    C_lists = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        def compute_process(gamma):
            return inversion(H_list, gamma)
        futures = [executor.submit(compute_process, gamma) for gamma in epsilon]
        concurrent.futures.wait(futures)
        for future in futures:
            C_lists.append(future.result())
    return C_lists


# 计算优度方式
def optimization(H_list, C_list):
    H_array = np.array(H_list)
    C_array = np.array(C_list)
    B = np.abs(np.matmul(H_array, C_array))
    # 求解B行列式
    detB = []
    for i in range(B.shape[0]):
        detB.append(np.linalg.det(B[i] - np.eye(M)))
    return sum(detB)


H_list = transport()
start = time.time()
C_lists = inversion1(H_list, [0.05, 0.02, 0.01, 0.04])
print(time.time() - start)