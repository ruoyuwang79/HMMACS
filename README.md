# RDLMA
**Based on DLMA, generalize to multiple-node, intelligent-sink, or underwater wireless communication cases.**

## How to Use

### Prerequisite

- NumPy (most of data are maintained as np.array)

- TensorFlow (DQN agent)
- Matplotlib (for mobility animation)

- In the ```src/``` directory, the ```run_{*}.py``` is the running script. Currently, the ```run_DQN.py``` is the simulation of intelligent-source.

## Update Logs

### Mobility Update

- Allow mobility of sources nodes

- User can use the self-designed track function to control nodes

- The track function is velocity driven, input current velocity, output should be next step velocity

- Demo

  ![demo](https://github.com/AliceEva/RDLMA/blob/main/figs/demo.gif)

## Note

- To use with the option ```save_trace=True``` or run ```plot_reward.py```,  you need to specify the proper storage paths. Recommend use ```configs/```, ```logs/```, and ```figs/```.
