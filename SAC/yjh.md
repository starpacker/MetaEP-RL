# 如果想要更改 参数自由度
1. env_core_revised.py 
2. train.py obs_dim_global/action_dim_global

关于envs中的collect data code:
judge_flag : 是否采用world model
multi: 是否探索多波长EP的结构
online_collect_data_4.py 是 online learning
multi_collect_data_3.py 是 pretrain model
