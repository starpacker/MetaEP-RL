#  实现eval功能，画图测试agent水平 (已经实现画图测试第一个agent的水平) 
#  找到train的完整流程，搞清楚input和output的类型和内容，以及可以调整的参数位置
#  调整reward  在collect_data_2.py里面的reward函数


# 在utils文件夹里面增加了draw.py, 
# 可以通过调用 draw.py 的 draw_ep_stru_phase_amplitude(stru, target= 100)

# 在env_runner.py里面，可以改变是否进行画图
observation [[[ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]
  [ 84  72  52 260  64  56  92 224  72  76]]]
reward [[0.41166361 0.41166361 0.41166361 0.41166361 0.41166361 0.41166361
  0.41166361 0.41166361 0.41166361 0.41166361]]
dones [[False False False False False False False False False False]]
information [[{} {} {} {} {} {} {} {} {} {}]]

这里是对于多智能体/单智能体的代码需要改的地方
1. train.py  parse_args   num_agents
2. env_core.py  init 初始化中action_dim 和 num_agents 都需要调整  (跑MAPPO的时候，用env_core,PPO的时候，用env_core_revised) self.agent_num,action_dim
3. algorithms ACTlayer  BOX action_dim = 9 / 4 
4. 在env_continuous.py里面切换
# from envs.env_core import EnvCore  MAPPO
# from envs.env_core_revised import EnvCore PPO
这里直接改了util.py里面的act_shape


judge还需要做进一步改进，对于高CD的预测还是不奏效
需要调整数据比例和损失函数

为了方便调整固定参数，重新写一个env_core_revised.py
并且将env_discrete.py & env_continuous.py 中的import进行了修改
通过调整 flag_env 来决定哪种运行方式

judge 目前需要做的几个实验： 
1. 不同threshold 进行对比
2. 不同网络的judge进行对比
3. 有无judge 进行对比

无judger: 
1 0.119791
2 0.695532
4 2.592406
9 2.952642
37 14.426152
712 25.788014
1529 70.157002
1717 79.872770

有judger(thre = 0.7) fcn
1 2.9
2 12.9
15 16.1
32 24.9
76 76.8
610 78.2
613 79.9

transformer:
1 2.979874
2 8.691383
4 9.200817
14 10.257494
17 64.054234
195 79.823857
265 94.41233

unet:
1 0.759629
2 2.880372
3 3.874581
8 4.458632
13 10.555228
22 25.892820
103 70.157004
395 79.872770
533 79.872780

不同threshold 对比： 
0.5
0.6
0.7
0.8

online learning judger:
thre = 0.6
1 4.216890
5 9.286607
9 9.367010
16 24.641757
25 67.493473
101 97.019371


thre = 0.7
1 2.979874
2 8.691383
4 9.200817
14 10.257494
17 64.054234




MAPPO thre = 0.7
reward_history_5

multi reward_history_6
0.7  reward_history_4
0.2  reward_history_0.2
PPO + BR: 直接利用大的batch解决就好 reward_history_2


Best Parameters: [128  68  52 152  48  48  76  92  80  76]
Best Reward: 33.539558
CD1: 0.030206, CD2: 0.812417
M1: 0.069461, M2: 0.055874
Pos1: 28, Pos2: 130

Best Parameters: [ 76  76  52 152  60  48  72  84  80  76]
Best Reward: 39.616521
CD1: 0.053455, CD2: 0.722416
M1: 0.115328, M2: 0.066274
Pos1: 53, Pos2: 74
                  
Best Parameters: [ 80  60  52 152  48  48 104  88  80  76]
Best Reward: 287.754882
CD1: 0.772775, CD2: 0.371071
M1: 0.089841, M2: 0.086746
Pos1: 86, Pos2: 132

Best Parameters: [ 44  56  60 144  60  48 128  92  72  76]
Best Reward: 24.589677
CD1: 0.061627, CD2: 0.382781
M1: 0.100451, M2: 0.088469
Pos1: 19, Pos2: 157

[ 88 , 56 , 52,144 , 52 , 40, 104 , 88 , 80 , 76]
[ 84  ,56,  56 ,148,  52 , 40, 104 , 88,  80,  76] 

[ 80,  60,  56, 148,  48,  40, 108,  84 , 76,  76]
[ 80,  56,  52, 144,  48 , 44, 108 , 92,  80,  76]

[ 88,  52,  52, 148,  48,  40, 108,  88,  76,  76]
[ 88,  56,  56, 144,  52,  44, 108,  88,  80,  76]


[ 92  52  60 144  60  48  92 248  72  76]

 [124  68  60 144  60  48  40  96  72  76]

  [112  60  76 152  60  48 120  88  72  50]

  [112  48  52 152  60  48 112  88  80  76]