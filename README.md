https://arxiv.org/abs/2007.04578

*Readme was written by Dongjae Kim (dongjaekim@dankook.ac.kr)*

# 1. Policy matching 

## 1.1. PFC-RL (./pfc-RL/)

*Codes were written by Dongjae Kim (dongjaekim@dankook.ac.kr)*

Two PFC-RL models are pre-trained as in the previous research (Lee et al., 2014). Since the models require 4 or 6 pre-trained parameters (unlike neural network's large number of weights), we just simply save it in csv formant for efficiency (regdata_*.csv files).

To generate simulation data of pfc-RL model1:

```
python main_gen_simul_task5.py --model-id=2  task-id=%d --sub-id=%d --GRW=%d --tree=%d
```

To generate simulation data of pfc-RL model2:

```
python main_gen_simul_task5.py --model-id=3  task-id=%d --sub-id=%d --GRW=%d --tree=%d
```

## 1.2. metaRL (./meta-RL/)

*Codes were written by  Minsu Abel Yang and Dongjae Kim (dongjaekim@dankook.ac.kr)*

We have uploaded the trained weights of PM-metaRL (./meta-RL/PM/SUBFIT). 

To generate simulation data of metaRL, simply run:

```
python main_gen_simul_task_meta_RL_PM.py --task-id=%d --sub-id=%d --GRW=%d --tree=%d
```

## 1.3. DDQN (./ddqn)

*Codes were written by  Jaehoon Shin (skalclrptsp@kaist.ac.kr)*

We have uploaded the trained weights of PM-metaRL (./meta-RL/PM/SUBFIT). 

To generate simulation data of metaRL, simply run:

```
python main_gen_simul_task_sbjtv_dqn.py --bhv_pseudo=0 --no_fitting=1 --fix_behavior=0 --task-id=%d --model-id=%d --sub-id=%d --GRW=%d --tree=%d  --episodes=%d 
```

## 1.4. Successor representation (./sr/)

*Codes were checked by Dongjae Kim (dongjaekim@dankook.ac.kr)*

```
python run_pm.py %d 0
```
where `%d` is the subject id in this model.

## 1.5. Implicit quantile network (./iqn/)

*Codes were checked by Dongjae Kim (dongjaekim@dankook.ac.kr)*

```
# training 
python run_pm.py --subid=%d --numsim=%d
# generating data
python main_generate_pm.py --subid=%d --numsim=%d --numiter=%d
```


Note that it is easy to convert from policy matching to behavior cloning since `behavioral cloning` is training model by rewarding (+1) the agent when commit the same action as the human subject

# 2. Goal matching

## 2.1. metaRL (./meta-RL/)

*Codes are written by  Minsu Abel Yang and Dongjae Kim (dongjaekim@dankook.ac.kr)*

We have uploaded the trained weights of PM-metaRL (./meta-RL/PM/SUBFIT). 

To generate simulation data of metaRL, simply run:

```
python main_gen_simul_task_meta_RL_GM.py --task-id=%d --sub-id=%d --GRW=%d --tree=%d
```

## 2.2. DDQN (./ddqn)

*Codes were written by  Jaehoon Shin (skalclrptsp@kaist.ac.kr)*

We have uploaded the trained weights of PM-metaRL (./meta-RL/PM/SUBFIT). 

To generate simulation data of metaRL, simply run:

```
python main_gen_simul_task_sbjtv_dqn.py --bhv_pseudo=1 --no_fitting=1 --fix_behavior=0 --task-id=%d --model-id=%d --sub-id=%d --GRW=%d --tree=%d  --episodes=%d 
```

## 2.3. Successor representation (./sr/)

*Codes were checked by Dongjae Kim (dongjaekim@dankook.ac.kr)*

```
python run_gm.py %d 0
```
`%d` represents the subject id in this model.

## 2.4. IQN (./iqn)

*Codes were checked by Dongjae Kim (dongjaekim@dankook.ac.kr)*

```
# training 
python run_gm.py --subid=%d --numsim=%d
# generating data
python main_generate_gm.py --subid=%d --numsim=%d --numiter=%d
```

## 

# 3. Generative Adversarial Imitation Learning (GAIL) (./gail)

*Codes were checked by Dongjae Kim (dongjaekim@dankook.ac.kr)*

```
# training 
python run.py --subid=%d --numiter=%d

# generating simulations
python generates.py --subid=%d --numsim=%d --numiter=%d
```

# 4. Arguments for 10 Markov decision tasks

Description of arguments to specify tasks.

```
--sub-id: subject id. We used 82 subjects paramter sets so  (0 to 81)
--model-id: Type of agent models. ex) 4: DDQN to maximize agents' rewards. 5: DDQN to copy the behavior of agents. 
--bhv_pseudo: The sequence of goal conditions in the trainig. 0: Do not use pseuod sequence and use real subejcts sequence (model-id=5), 1: Do the pseudo sequence with equal numbers of each scenario (mode-id=4)
--no-fitting: Update DDQN or fix DDQN. 0: Update agents. 1: Freeze agents
--episodes: Number of episodes in training
--fix-behavior: The computational agents' sequences of visited states are exactly same as subjects' real data or not 0: Agents move as the agents select. 1: Agents move as the human subject's history.
--task-id : Type of tasks (1~10). ## the order is different from the order on the paper. please check arguments before running codes.
--GRW: the method to control state transition uncertainty using Gaussian random walk process or not. 0: switch-switch model, 1: fixed-drift, 2: drift-drift 3: fixed-(drif+switch) 4: (drift+switch)-(drift+switch)
--tree: Tree or ladder shape of the task 0: ladder, 1: tree
```
