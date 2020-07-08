python main_gen_simul_task_sbjtv_dqn.py --task-id=%d --model-id=%d --sub-id=%d --GRW=%d --tree=%d --bhv_pseudo=%d --no_fitting=%d --episodes=%d --fix_behavior=%d

%d : Arguments

*Result

Store as .npy file in the simul_results/ Path

*Arguments 
--sub-id: subject id. We used 82 subjects paramter sets so  %d becomes 0~81
--model-id: Type of agent models. 4: DDQN to maximize agents' rewards. 5: DDQN to copy the behavior of agents. 
--bhv_pseudo: The sequence of goal conditions in the trainig. 0: Do not use pseuod sequence and use real subejcts sequence (model-id=5), 1: Do the pseudo sequence with equal numbers of each scenario (mode-id=4)
--no-fitting: Update DDQN or fix DDQN. 0: Update agents. 1: Freeze agents
--episodes: Number of episodes in training
--fix-behavior: The computational agents' sequences of visited states are exactly same as subjects' real data or not 0: Agents move as the agents select. 1: Agents move as the human subject's history.
--task-id : Type of tasks. 1~10. follow *10 task environments
--GRW: Gaussian random walk property. 0: drift model, 1: payoff only, 2: all state-transition 3:Drift-shift in the payoff only 4: Drift-shift in the all state-transition
--tree: Tree or ladder shape of the task 0: ladder, 1: tree

* 10 task environments

'--task-id=1 --GRW=2 --tree=1'

'--task-id=2 --GRW=1 --tree=1'

'--task-id=3 --GRW=0 --tree=1'   *** original 2014 task

'--task-id=4 --GRW=2 --tree=0'

'--task-id=5 --GRW=1 --tree=0'

'--task-id=6 --GRW=0 --tree=0'

'--task-id=7 --GRW=3 --tree=1'

'--task-id=8 --GRW=4 --tree=1'

'--task-id=9 --GRW=3 --tree=0'

'--task-id=10 --GRW=4 --tree=0'