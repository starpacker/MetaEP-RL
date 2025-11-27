import random
import envs.package as pkg
import numpy as np
import os, time

out_of_bound_reward = -0.5
best_rewards = 0.0
flag_2 = True

# 新增记录 best reward 和对应 simulation 次数的列表
best_records = []
simulation_count = 0  # 记录调用 runlume 的次数

def runlume_8(
    stru_name='pra_8.txt',
    data_name="farfile_reflection_8.txt",
    model_file="jones_model_origin_8.fsp",
    mother_script="cal_farfield_data.lsf",
):
    def add_quota(string):
        return '\"' + string + '\"'
    
    file_root = "C:/data"
    stru_name = os.path.join(file_root, stru_name)
    data_name = os.path.join(file_root, data_name)
    model = os.path.join(file_root, model_file)
    fdtd_solutions = 'C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe'
    fsp_file = model
    lsf_file = "C:/data/script_8.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
    os.system(cmd)
    return True

CD_global_best = 0
rl_lr_global_best = 0
m_global_best = np.inf
time_intervent = 0


def reward_8(data):
    global best_rewards

    # wave_length = np.array(data['wavelength'])
    # eig_state_1_real = np.array(data['eig_state_1_real'])
    # eig_state_1_imag = np.array(data['eig_state_1_imag'])
    # eig_state_2_real = np.array(data['eig_state_2_real'])
    # eig_state_2_imag = np.array(data['eig_state_2_imag'])
    r_lr_real = np.array(data['r_lr_real'])
    r_lr_imag = np.array(data['r_lr_imag'])
    r_rl_real = np.array(data['r_rl_real'])
    r_rl_imag = np.array(data['r_rl_imag'])
    r_rr_real = np.array(data['r_rr_real'])
    r_rr_imag = np.array(data['r_rr_imag'])
    r_lr = r_lr_real**2 + r_lr_imag**2
    r_rl = r_rl_real**2 + r_rl_imag**2
    r_rr = r_rr_real**2 + r_rr_imag**2
    cd_sequence = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr)

    # 寻找所有峰值：上升到顶点 -> 下降到谷底 -> 再上升 -> 记录新峰值...
    peaks = []  # 存储 (index, value)
    n = len(cd_sequence)

    i = 1
    while i < n - 1:

        # 上升阶段
        while i < n - 1 and cd_sequence[i] <= cd_sequence[i + 1]:
            i += 1

        # 此时 i 可能是峰值（局部最大）
        if i > 0 and i < n - 1 and cd_sequence[i] > cd_sequence[i - 1] and cd_sequence[i] >= cd_sequence[i + 1]:
            # 检查是否严格大于邻居（允许平台，但取平台最左）
            # 如果后续是平台，跳过平台
            peak_index = i
            peak_value = cd_sequence[i]
            # 跳过平台顶部
            while i < n - 1 and cd_sequence[i] == cd_sequence[i + 1]:
                i += 1
            peaks.append((peak_index, peak_value))

        # 下降阶段
        while i < n - 1 and cd_sequence[i] >= cd_sequence[i + 1]:
            i += 1

    # 按峰值数量设计奖励
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)  # 按CD值降序

    if len(peaks) >= 3:
        # 额外记录到日志文件
        with open("C:\\data\\multi_peak_log_8.txt", 'a') as logf:
            logf.write("="*50 + "\n")
            logf.write(f"Parameters: {data['pra']}\n")
            logf.write(f"Found {len(peaks)} peaks (top 3: {[(idx, val) for idx, val in peaks_sorted[:3]]})\n")
            logf.write(f"All peaks: {[(idx, f'{val:.4f}') for idx, val in peaks]}\n")
            logf.write(f"Reward computed from top 2: {peaks_sorted[0][1]:.4f}, {peaks_sorted[1][1]:.4f}\n")

    # 奖励计算：始终使用最大的两个峰值（如果存在）
    if len(peaks_sorted) == 1:
        cd1 = peaks_sorted[0][1]
        cd2 = 0.0
        base_reward = 1.0
        quality_bonus = cd1 * 2 # 单峰加成较低
    else:
        cd1 = peaks_sorted[0][1]
        cd2 = peaks_sorted[1][1]
        base_reward = 2.0  # 双峰及以上给更高基础奖励
        quality_bonus = (cd1 + cd2) * 2 + cd1 * cd2 * 10   # 双峰乘积放大

    total_reward = base_reward + quality_bonus

    # 更新最佳结果（保持原逻辑）
    if total_reward > best_rewards:
        best_rewards = total_reward
        with open("C:\\data\\best_result_8.txt", 'w') as f:
            f.write(f"Best Parameters: {data['pra']}\n")
            f.write(f"Best Reward: {best_rewards:.6f}\n")
            f.write(f"Top CD1: {cd1:.6f}, CD2: {cd2:.6f}\n")
            f.write(f"Number of Peaks: {len(peaks)}\n")
            f.write(f"All Peak Positions: {[p[0] for p in peaks]}\n")
            f.write(f"All Peak Values: {[f'{p[1]:.6f}' for p in peaks]}\n")

    return total_reward

def run_onetime_8(pra):
    global simulation_count
    filepra = ','.join(str(p) for p in pra)
   
    filepath = os.path.join("C:\\data\\pra", filepra) + '.txt'
    pra = pra[:10]
    if os.path.exists(filepath):
        data = np.loadtxt(filepath)
    else:
        np.savetxt('C:\\data\\pra_8.txt', pra)
        runlume_8()
        if not os.path.exists("C:\\data\\farfile_reflection_8.txt"):
            simulation_count += 1 
            return out_of_bound_reward
        data = np.loadtxt("C:\\data\\farfile_reflection_8.txt")
        os.remove("C:\\data\\farfile_reflection_8.txt")  # 删除文件

    simulation_count += 1 
    pattern = pkg.PatternRects(pra)
    stru = pattern.get_details(op=data)
    if stru is None:
        return out_of_bound_reward
    
    return reward_8(stru)

# -------------------------------------------- #
# judge_flag用于实现判别器
judge_flag = False
# judge_flag = True
if judge_flag:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # from judge.CD_inference import CustomResUNet
    # judger_model = CustomResUNet().to(device)
    # judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_0719.pth"))
    # judger_model.eval()

    # from judge.FCN import FCNRegressor
    # judger_model = FCNRegressor().to(device)
    # judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_fcn_0729.pth"))
    # judger_model.eval()

    from judge.whole_trans import EdgeLengthTransformerRegressor
    judger_model = EdgeLengthTransformerRegressor().to(device)
    judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_trans_full_sequence_0904.pth"))
    judger_model.eval()
    # 参数边界定义
    param_bounds = {
        0: (30, 160),
        1: (40, 100),
        2: (40, 80),
        3: (80, 340),
        4: (40, 100),
        5: (40, 100),
        6: (30, 150),
        7: (40, 340),
        8: (40, 100),
        9: (30, 100)
    }

    def normalize(params):
        """
        将参数归一化到[0,1]范围
        params: [N, 10] 或 [10] (支持 numpy 数组或 torch Tensor，包括 CUDA Tensor)
        """
        # 检测输入类型并转换为 numpy 数组
        if isinstance(params, torch.Tensor):
            # 如果是 CUDA Tensor，先转移到 CPU 再转为 numpy
            if params.is_cuda:
                params = params.cpu()
            params = params.detach().numpy()
        else:
            params = np.array(params)
        original_shape = params.shape
        if len(original_shape) == 1:
            params = params.reshape(1, -1)
        
        normalized = np.zeros_like(params, dtype=np.float32)
        for i in range(len(param_bounds)):
            min_val, max_val = param_bounds[i]
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
            normalized[:, i] = np.clip(normalized[:, i], 0, 1)
        
        return torch.from_numpy(normalized).float().to(device)

def get_pattern_8(pra):  # ✅ 修正为 get_pattern_8
    pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10 = pra
    pattern = np.zeros((400, 400), dtype=int)
    for xi in range(pattern.shape[0]):
        for yj in range(pattern.shape[1]):
            if xi >= pra1 and xi <= pra1 + pra3:
                if yj >= pra10 + pra9 + pra6 + pra5 and \
                        yj <= pra10 + pra9 + pra6 + pra5 + pra2:
                    pattern[xi, yj] = 1
            if xi >= pra1 and xi <= pra1 + pra4:
                if yj >= pra10 + pra9 + pra6 and \
                        yj <= pra10 + pra9 + pra6 + pra5:
                    pattern[xi, yj] = 1
            if xi >= pra7 and xi <= pra7 + pra8:
                if yj >= pra10 and yj <= pra10 + pra9:
                    pattern[xi, yj] = 1
    return pattern

def run_onetime_total_new_8(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
    global flag_2
    global simulation_count
    pra_short = pra[:10]  

    # 生成 pattern
    pattern = get_pattern_8(pra_short)  # ✅ 修正函数名
    # 转换为模型输入格式 (B, C, H, W)
    input_tensor = torch.from_numpy(pattern).float().unsqueeze(0).unsqueeze(0).to(device)

    # 使用 judger 模型进行预判
    with torch.no_grad():
        predicted_reward = judger_model(input_tensor).item()
        
    # 设置阈值（根据模型输出范围设定）
    threshold = 0.40
    if predicted_reward < threshold:
        return predicted_reward + 1  
    print("judger:", predicted_reward)

    #否则继续执行 FDTD 模拟
    np.savetxt('C:\\data\\pra_8.txt', pra_short)
    runlume_8()
    if not os.path.exists("C:\\data\\farfile_reflection_8.txt"): 
        simulation_count += 1 
        return out_of_bound_reward
    
    simulation_count += 1   

    data = np.loadtxt("C:\\data\\farfile_reflection_8.txt")
    os.remove("C:\\data\\farfile_reflection_8.txt")  # 删除文件

    flag_2 = True

    pattern_obj = pkg.PatternRects(pra_short)
    stru = pattern_obj.get_details(op=data)

    if stru is None:
        return out_of_bound_reward

    return reward_8(stru)

# def run_onetime_trans(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
#     global flag_2
#     global simulation_count
#     pra_short = pra[:10]  

#     input_tensor = torch.tensor(pra_short, dtype=torch.float32).unsqueeze(0).to(device)
#     with torch.no_grad():
        
#         tensor_value = judger_model(normalize(input_tensor)).cpu()
#     # print(tensor_value.shape)

#     pos1, pos2 = find_two_positions_by_argmin(tensor_value.squeeze(0), threshold=0.1, min_distance=20)
#     if pos1 == None: 
#         return 0.1
#     print(f"judger:{pos1},{pos2}")
    

#     np.savetxt('C:\\data\\pra_8.txt', pra_short)
#     runlume_8()
#     if not os.path.exists("C:\\data\\farfile_reflection_8.txt"): 
#         simulation_count += 1 
#         return out_of_bound_reward
    
#     simulation_count += 1   

#     data = np.loadtxt("C:\\data\\farfile_reflection_8.txt")
#     os.remove("C:\\data\\farfile_reflection_8.txt")

#     flag_2 = True

#     pattern_obj = pkg.PatternRects(pra_short)
#     stru = pattern_obj.get_details(op=data)

#     if stru is None:
#         return out_of_bound_reward

#     return reward_8(stru)

def get_reward():
    reward_all = []
    pras = np.loadtxt("C:\\data\\pra_all_8.txt")
    pra_int = pras.astype(int)
    if pra_int[0] == 0:
        reward_all.append(out_of_bound_reward)
    else:
        if not judge_flag:
            reward_all.append(run_onetime_8(pra_int))

    with open("C://data//reward_history_8.txt", 'a') as file:
        for w in reward_all:
            file.write(f"{pra_int}\n{w}\n")
    np.savetxt("C:\\data\\reward_8.txt", np.array(reward_all))