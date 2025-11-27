切换多智能体/单智能体的代码需要改的地方
1. train.py  parse_args: num_agents
2. env_continuous.py  跑MAPPO的时候，用env_core;PPO的时候，用env_core_revised;
注意调整代码中的 self.agent_num,action_dim
3. algorithms ACTlayer  BOX action_dim 

python train/train.py
会生成如下数据:
1.CD_history.txt
2.reward_history.txt
3.best_result.txt
4.record.txt  (record how the best result changes)

调整参数在config.py里