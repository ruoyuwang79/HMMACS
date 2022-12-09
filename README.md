# Hybrid Mobile MAC Simulator (HMMACS)
**A light-weighted, user-friendly, AI-integratable, and mobility-compatible MAC simulator.**

## How to Use

### Prerequisite

The versions are based on the development environment, lower version maybe workable, but no guarantee.

- Python ```>=3.8.10```

- NumPy  ```>=1.21.1``` (most of data are maintained as ```np.array```)
- TensorFlow ```>=2.9.1+nv22.8``` (DQN agent)
- Matplotlib ```>=3.5.0``` (for mobility animation)

Or you can use the docker image we used during development:

```
docker run --gpus all -it --rm nvcr.io/nvidia/tensorflow:22.11-tf2-py3
```



### Usage

```
usage: main.py [-h] 
[--n_agents N] [--n_others N] [--packet_length N] [--guard_length N] [--sub_slot_length T] [--delay_max N] 
[--num_sub_slot N] [--frame_length N] [--tdma_occupancy N] [--aloha_prob q] [--window_size N] [--max_backoff N] 
[--env_mode M] [--env_mac_mode M] [--agent_mac_mode M] [--sink_mode M] [--state_len M] [--memory_size E] 
[--replace_target_iter F] [--train_interval F] [--batch_size B] [--gamma G] [--alpha A] [--lr LR] [--reward_polarity] 
[--penalty_factor P] [--epsilon e] [--epsilon_min MIN] [--epsilon_decay D] [--movable] [--time_granularity T] 
[--distance_init] [--random_init] [--save_trace] [--save_track] [--max_iter N] [--log_path DIR] [--config_path DIR] 
[--track_path DIR] [--file_prefix NAME] [--setup_path DIR] [--mask NAME] [--delay NAME] [--x NAME] [--y NAME] [--z NAME]
```

The description of those options are:

```
optional arguments:
  -h, --help            show this help message and exit
  --n_agents N          number of agents (default 0)
  --n_others N          number of non-agents (default 2)
  --packet_length N     packet length (default 3 sub slots)
  --guard_length N      guard length (default 3 sub slots)
  --sub_slot_length T   length of sub time slot (default 1e7 ns)
  --delay_max N         maximum delay, or distance upper bound (default 100 sub slots)
  --num_sub_slot N      number of sub time slot per time slot (default 20 sub slots
  --frame_length N      length of a TDMA time frame (default 10 time slots)
  --tdma_occupancy N    number of TDMA transmissions in a frame (default 3 times)
  --aloha_prob q        ALOHA transmission probability q (default 0.5)
  --window_size N       ALOHA fixed window or exponential backoff window initial window size (default 2 time slots)
  --max_backoff N       ALOHA exponential backoff maximum backoff (default 2)
  --env_mode M          medium selection, 0 RF, others UAN (default 1)
  --env_mac_mode M      inbuilt MAC mode, 0 sync, others async (default 1)
  --agent_mac_mode M    agent MAC mode, 0 sync, others async (default 1)
  --sink_mode M         agent node type, 0 source node, 1 sink node (default 0)
  --state_len M         length of super-state (default 20 time slots)
  --memory_size E       size of history state memory (default 1000 super-states)
  --replace_target_iter F
                        update frequency of target network (default 20 time slots)
  --train_interval F    training frequency (default 5 time slots)
  --batch_size B        training batch size (default 64)
  --gamma G             discount factor gamma (default 0.9)
  --alpha A             Bellman equation learning rate alpha (default 1.0)
  --lr LR               NN optimizer learning rate (default: 0.01)
  --reward_polarity     if set this flag, the DQN will have a non-zero penalty for the collision
  --penalty_factor P    non-zero penalty scale (default 1.0)
  --epsilon e           epsilon decay start value (default 1)
  --epsilon_min MIN     epsilon decay minimum value (default 0.01)
  --epsilon_decay D     epsilon decay factor (default 0.995)
  --movable             if set this flag, the simulator will update delays during runtime
  --time_granularity T  time period of spatial simulator (default 30e9 ns)
  --distance_init       if set this flag, the simulator will use the delay to initialize position
  --random_init         if set this flag, the simulator will randomly initialize position
  --save_trace          if set this flag, the simulator will save throughput
  --save_track          if set this flag, the simulator will save moving track
  --max_iter N          number of iterations (default 50000 time slots)
  --log_path DIR        log (throughput) path to use (default: ../logs/)
  --config_path DIR     config (experiment setup) path to use (default: ../configs/)
  --track_path DIR      track (moving positions) path to use (default: ../tracks/)
  --file_prefix NAME    file prefix (default: None)
  --setup_path DIR      setup path (default: ../setups/)
  --mask NAME           mask file name (default: None)
  --delay NAME          delay file name (default: None)
  --x NAME              x position file name (default: None)
  --y NAME              y position file name (default: None)
  --z NAME              z position file name (default: None)
```

Or you can use the running script ```run.sh``` to launch batches of experiments simultaneously.

```
./run.sh
```

Note that the ```run.sh``` exists in the root directory of the repository, the directory of IO files should be different with the default value.



## Update Logs

### Mobility Update

- Allow mobility of sources nodes

- User can use the self-designed track function to control nodes

- The track function is velocity driven, input current velocity, output should be next step velocity

- Demo

  ![demo](https://github.com/AliceEva/RDLMA/blob/main/figs/demo.gif)

## Note

- To use with the option ```save_trace=True``` or run ```plot_reward.py```,  you need to specify the proper storage paths. Recommend use ```configs/```, ```logs/```, and ```figs/```.
