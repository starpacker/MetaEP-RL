from envs.best_response import *
TARGET_SIM = 2000
total_sim = 0
search_indices = [0, 1, 6, 7]

import random

print("==== 启动 Best Response 迭代优化（目标仿真次数 2000）====")

# ------------------------------
# Step 1: 随机生成一个合法初始参数 pra
# ------------------------------
while True:
    pra = [None, None, 52, 152, 48, 48, None, None, 80, 76]

    pra[0] = random.randint(30, 160)
    pra[1] = random.randint(40, 100)
    pra[6] = random.randint(30, 150)
    pra[7] = random.randint(40, 340)

    pra_arr = np.array(pra)
    if boundary_test(pra_arr):
        break

print(f"\n==== 初始随机 pra: {pra} ====\n")

# ------------------------------
# Step 2: 开始循环调用 run_best_response_search
# ------------------------------
current_pra = pra.copy()

while total_sim < TARGET_SIM:

    print("\n========================")
    print(" 开始新一轮 Best Response")
    print("========================\n")

    result = run_best_response_search(current_pra, search_indices)

    sim_used = result["simulation_count"]
    best_pra = result["final_pra"]

    total_sim += sim_used

    print(f"\n>>> 本轮仿真次数： {sim_used}")
    print(f">>> 当前累计仿真： {total_sim}/{TARGET_SIM}")
    print(f">>> 本轮最优参数： {best_pra}\n")

    # 使用本轮最优解作为下一轮起点
    current_pra = best_pra.copy()

print("\n==== 达到 2000 次仿真上限，优化结束 ====")
print(f"最终参数： {current_pra}")
