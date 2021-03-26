# WumpusWorld-RL
Reinforcement learning in WumpusWorld

Project created with Anaconda.  
Python version: 3.8.5  
Additional packages installation:
- Matplotlib: conda install -c anaconda matplotlib
- Pytorch CPU: conda install pytorch torchvision torchaudio cpuonly -c pytorch  
More help: https://pytorch.org  

# Usage
<pre>
python main.py [-h] --env ENV --mode MODE [--num_episodes NUM_EPISODES]
               [--max_steps_per_episode MAX_STEPS_PER_EPISODE] [--lr LR] [--discount DISCOUNT]
               [--eps_start EPS_START] [--eps_decay EPS_DECAY] [--eps_min EPS_MIN]
               [--show_actions_plot SHOW_ACTIONS_PLOT] [--show_reward_plot SHOW_REWARD_PLOT]
               [--show_games_won_plot SHOW_GAMES_WON_PLOT] [--show_learned_path SHOW_LEARNED_PATH]
               [--batch_size BATCH_SIZE] [--target_update TARGET_UPDATE]
               [--memory_size MEMORY_SIZE]  
               
               
optional arguments:
  -h, --help                                    show this help message and exit  
  --num_episodes NUM_EPISODES                   Number of learning episodes, default=10000  
  --max_steps_per_episode MAX_STEPS_PER_EPISODE Maximum number of steps per episode, default=100  
  --lr LR                                       Learning rate, should be in <0, 1>, default=0.1  
  --discount DISCOUNT                           Discount rate (gamma), should be in <0, 1>, default=0.9  
  --eps_start EPS_START                         Epsilon starting value, should be in <0, 1>, default=1  
  --eps_decay EPS_DECAY                         Epsilon decay rate, default=0.001  
  --eps_min EPS_MIN                             Epsilon min, should be in <0, 1>, default=0.01  
  --show_actions_plot SHOW_ACTIONS_PLOT         Show plot with number of actions, default=True  
  --show_reward_plot SHOW_REWARD_PLOT           Show plot with rewards, default=True  
  --show_games_won_plot SHOW_GAMES_WON_PLOT     Show plot with games won, default=True  
  --show_learned_path SHOW_LEARNED_PATH         Show learned path after learning, default=True  
  --batch_size BATCH_SIZE                       Used in DQN replay memory, default=256  
  --target_update TARGET_UPDATE                 Used in DQN, tells how often target network should be
                                                updated, default=10  
  --memory_size MEMORY_SIZE                     Used in DQN, set replay memory size, default=100000  

required named arguments:  
  --env ENV             Required, choose environment, possible values: lv1, lv2, lv3v1, lv3v2, lv3v3, lv4  
  --mode MODE           Required, choose mode, possible values: manual, q-learn, dqn  
</pre>
