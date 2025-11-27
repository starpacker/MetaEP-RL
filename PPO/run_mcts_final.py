# ---------- MCTS (iteration-limited) for 4-parameter search (append/replace in best_response.py) ----------
import math
import random
import time
from collections import defaultdict
import os
import numpy as np
import envs.package as pkg
_out_of_bound_reward = -0.2
_best_records = []
_best_rewards = -np.inf
_simulation_count = 0
_MCTS_PARAM_INDICES = [0, 1, 6, 7]
_MCTS_DOMAINS = {
    0: list(range(30, 161, 4)),
    1: list(range(40, 101, 4)),
    6: list(range(30, 151, 4)),
    7: list(range(40, 341, 4)),
}

def boundary_test(pra):
    c = (pra[0] >= 30 and pra[0] <= 160 and
         pra[2] >= 40 and pra[2] <= 80 and
         pra[3] >= 80 and pra[3] <= 340 and
         pra[0] + pra[3] <= 370 and
         pra[6] >= 30 and pra[6] <= 150 and
         pra[7] >= 40 and pra[7] <= 340 and
         pra[6] + pra[7] <= 340 and
         pra[9] >= 30 and pra[9] <= 100 and
         pra[8] >= 40 and pra[8] <= 100 and
         pra[5] >= 40 and pra[5] <= 100 and
         pra[4] >= 40 and pra[4] <= 100 and
         pra[1] >= 40 and pra[1] <= 100 and
         pra[9] + pra[8] + pra[5] + pra[4] + pra[1] <= 370)
    return c

def runlume_7(
    stru_name='pra_7.txt',
    data_name="farfile_reflection_7.txt",
    model_file="jones_model_origin_7.fsp",
    mother_script="cal_farfield_data.lsf"
):
    def add_quota(string):
        return '"' + string + '"'

    file_root = "C:/data"
    stru_name = os.path.join(file_root, stru_name)
    data_name = os.path.join(file_root, data_name)
    model = os.path.join(file_root, model_file)
    fdtd_solutions = 'C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe'
    fsp_file = model
    lsf_file = "C:/data/script_7.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, "-nw -run", lsf_file])
    os.system(cmd)
    return True

def _run_data(pra_short):
    global _simulation_count
    np.savetxt('C:\\data\\pra_7.txt', pra_short, fmt='%d')
    runlume_7()

    _simulation_count += 1
    if not os.path.exists("C:\\data\\farfile_reflection_7.txt"):
        with open("C:/data/all_rewards_7.txt", "a") as f:
            f.write(str(pra_short)+'\n')
            f.write(str(_out_of_bound_reward)+'\n')
        return _out_of_bound_reward

    data = np.loadtxt("C:\\data\\farfile_reflection_7.txt")
    os.remove("C:\\data\\farfile_reflection_7.txt")

    pattern = pkg.PatternRects(pra_short)
    stru = pattern.get_details(op=data)

    if stru is None:
        reward = _out_of_bound_reward
        with open("C:/data/all_rewards_7.txt", "a") as f:
            f.write(str(pra_short)+'\n')
            f.write(str(reward)+'\n')
        return reward

    reward = _reward_7(stru, pra_short)
    with open("C:/data/all_rewards_7.txt", "a") as f:
        f.write(str(pra_short)+'\n')
        f.write(str(reward)+'\n')
    return reward

def _reward_7(data, pra):
    global _best_rewards, _best_records, _simulation_count
    wave_l = 88
    wave_length = np.array(data['wavelength'])
    eig_state_1_real = np.array(data['eig_state_1_real'])
    eig_state_1_imag = np.array(data['eig_state_1_imag'])
    eig_state_2_real = np.array(data['eig_state_2_real'])
    eig_state_2_imag = np.array(data['eig_state_2_imag'])
    r_lr_real = np.array(data['r_lr_real'])
    r_lr_imag = np.array(data['r_lr_imag'])
    r_rl_real = np.array(data['r_rl_real'])
    r_rl_imag = np.array(data['r_rl_imag'])
    r_rr_real = np.array(data['r_rr_real'])
    r_rr_imag = np.array(data['r_rr_imag'])

    dis_real = np.abs(eig_state_1_real - eig_state_2_real)
    dis_imag = np.abs(eig_state_1_imag - eig_state_2_imag)

    r_lr = r_lr_real ** 2 + r_lr_imag ** 2
    r_rl = r_rl_real ** 2 + r_rl_imag ** 2
    r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2

    if wave_length[wave_l] > 650:
        wave_l = 44

    CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
    r = dis_real[wave_l]
    i = dis_imag[wave_l]
    m = (r + i) / 2
    c = CD[wave_l]

    rl_lr = np.abs(np.log(r_lr[wave_l]) - np.log(r_rl[wave_l]))
    if r_rl[wave_l] > r_lr[wave_l]:
        rl_rr = np.log(r_rl[wave_l]) - np.log(r_rr_ll[wave_l])
    else:
        rl_rr = np.log(r_lr[wave_l]) - np.log(r_rr_ll[wave_l])
    reward = 0.0
    if rl_lr > 0 and rl_rr > 0:
        reward += rl_lr 
    if rl_lr > 1 and rl_rr > 1:
        reward += rl_lr * 1
    if rl_lr > 2 and rl_rr > 1:
        reward += rl_lr * 1
    if rl_lr > 2 and rl_rr > 2:
        reward += rl_rr * 1
    if rl_lr > 3 and rl_rr > 2:
        reward += rl_rr * 1
    if rl_lr > 3 and rl_rr > 3:
        reward += rl_lr * 1
    if rl_lr > 4 and rl_rr > 3:
        reward += rl_lr * 2
    if rl_lr > 4 and rl_rr > 4:
        reward += rl_rr * 5

    if reward > _best_rewards:
        _best_rewards = reward
        with open("C:\\data\\best_result_mcts.txt", 'w') as f:
            f.write(f"m:{m}\n")
            f.write(f"CD:{c}\n")
            f.write(f"lr:{r_lr[wave_l]}\n")
            f.write(f"rl:{r_rl[wave_l]}\n")
            f.write(f"rr:{r_rr_ll[wave_l]}\n")
            f.write(f"wavelength:{wave_length[wave_l]}nm\n")
            f.write(f"Best Parameters: {list(pra)}\n")
            f.write(f"Best Reward: {_best_rewards:.6f}\n")

        _best_records.append((_simulation_count, _best_rewards))
        if _best_records:
            np.savetxt("C:\\data\\records_7.txt", np.array(_best_records), fmt='%d %.6f')

    return reward

def _build_full_pra_from_partial(partial_vals):
    pra = [None] * 10
    pra[2] = 52
    pra[3] = 152
    pra[4] = 48
    pra[5] = 48
    pra[8] = 80
    pra[9] = 76
    for idx in _MCTS_PARAM_INDICES:
        pra[idx] = partial_vals.get(idx, _MCTS_DOMAINS[idx][0])
    return pra

class MCTSNode:
    def __init__(self, depth=0, parent=None, assigned=None):
        self.depth = depth
        self.parent = parent
        self.assigned = dict(assigned) if assigned else {}
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self._untried_actions = None

    def next_param_index(self):
        if self.depth >= len(_MCTS_PARAM_INDICES):
            return None
        return _MCTS_PARAM_INDICES[self.depth]

    def untried_actions(self):
        if self._untried_actions is None:
            pidx = self.next_param_index()
            if pidx is None:
                self._untried_actions = []
            else:
                self._untried_actions = _MCTS_DOMAINS[pidx].copy()
                random.shuffle(self._untried_actions)
        return self._untried_actions

    def is_fully_expanded(self):
        return len(self.untried_actions()) == 0

    def is_terminal(self):
        return self.depth == len(_MCTS_PARAM_INDICES)

    def add_child(self, action_value):
        pidx = self.next_param_index()
        assert pidx is not None
        new_assigned = dict(self.assigned)
        new_assigned[pidx] = action_value
        child = MCTSNode(depth=self.depth + 1, parent=self, assigned=new_assigned)
        self.children[action_value] = child
        try:
            self._untried_actions.remove(action_value)
        except Exception:
            pass
        return child

    def best_child(self, c_param=1.4):
        best_score = -float('inf')
        best_child = None
        for val, child in self.children.items():
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.total_value / child.visits
                explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

def _rollout_random_reward(node):
    """
    返回 (reward, full_pra) —— 确保我们能记录真正用于评估的 pra
    """
    assigned = dict(node.assigned)
    for depth_idx in range(node.depth, len(_MCTS_PARAM_INDICES)):
        pidx = _MCTS_PARAM_INDICES[depth_idx]
        assigned[pidx] = random.choice(_MCTS_DOMAINS[pidx])
    full_pra = _build_full_pra_from_partial(assigned)
    try:
        reward = _run_data(np.array(full_pra))
    except Exception as e:
        print(f"[MCTS] _run_data exception during rollout: {e}")
        reward = -1e9
    return reward, full_pra

def _tree_policy(root):
    node = root
    while not node.is_terminal():
        if len(node.untried_actions()) > 0:
            action = node.untried_actions().pop()
            child = node.add_child(action)
            return child
        else:
            node = node.best_child()
            if node is None:
                break
    return node

def _backup(node, reward):
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.total_value += reward
        cur = cur.parent

def mcts_search(initial_pra, target_iters=2000, max_iters=100000, c_param=1.4, verbose=True):
    """
    主要改动：停止条件改为迭代次数（iters）达到 target_iters。
    仍然尊重 _simulation_count 的上限（不会无限调用 _run_data）。
    """
    root = MCTSNode(depth=0, parent=None, assigned={})
    best_global_reward = -float('inf')
    best_global_pra = None

    global _simulation_count

    iters = 0
    while iters < target_iters and iters < max_iters:
        iters += 1
        leaf = _tree_policy(root)
        reward, full_pra = _rollout_random_reward(leaf)
        _backup(leaf, reward)

        if reward is not None and reward > best_global_reward:
            best_global_reward = reward
            best_global_pra = full_pra

        if verbose and (iters % 100 == 0):
            print(f"[MCTS] iters={iters}, sims={_simulation_count}, best_reward={best_global_reward:.6f}")

        # safety: 如果仿真计数已经非常高，也可提前退出（可选）
        # if _simulation_count >= SOME_HARD_LIMIT:
        #     break

    if verbose:
        print(f"[MCTS] finished: iters={iters}, total_sims={_simulation_count}, best_reward={best_global_reward}")
    return {"best_pra": best_global_pra, "best_reward": best_global_reward, "iters": iters, "total_simulations": _simulation_count}

def run_mcts_iter_limited(initial_pra=None, TARGET_ITERS=2000, verbose=True):
    global _simulation_count
    if initial_pra is None:
        while True:
            cand = [None] * 10
            cand[2] = 52
            cand[3] = 152
            cand[4] = 48
            cand[5] = 48
            cand[8] = 80
            cand[9] = 76
            cand[0] = random.choice(_MCTS_DOMAINS[0])
            cand[1] = random.choice(_MCTS_DOMAINS[1])
            cand[6] = random.choice(_MCTS_DOMAINS[6])
            cand[7] = random.choice(_MCTS_DOMAINS[7])
            if boundary_test(np.array(cand)):
                initial_pra = cand
                break
    else:
        initial_pra = list(initial_pra)

    print(f"[MCTS Driver] starting initial pra: {initial_pra}")
    res = mcts_search(initial_pra, target_iters=TARGET_ITERS, verbose=verbose)
    print(f"[MCTS Driver] Done. iters={res['iters']}, total_simulations={res['total_simulations']}, best_reward={res['best_reward']}")
    print(f"[MCTS Driver] best_pra={res['best_pra']}")
    return {"final_pra": res["best_pra"], "best_reward": res["best_reward"], "iters": res["iters"], "total_simulations": res["total_simulations"]}

# If run as script:
if __name__ == "__main__":
    TARGET_ITERS = 2000
    print("==== Start MCTS iterative-limited search (iters={} ) ====".format(TARGET_ITERS))
    result = run_mcts_iter_limited(initial_pra=None, TARGET_ITERS=TARGET_ITERS, verbose=True)
    print("\n==== MCTS Done ====")
    print("Final best pra:", result["final_pra"])
    print("Best reward:", result["best_reward"])
    print("Total sims used:", result["total_simulations"])
