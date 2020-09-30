# Lyapunov Barrier Policy Optimization

In submission to ICLR 2021. This code reproduces the results for our method and baselines showed in the paper.

## Install
- PyTorch 1.5
- OpenAI Gym
- [MuJoCo](https://www.roboti.us/license.html)
- [OpenAI safety gym](https://github.com/openai/safety-gym)


## Instructions

- All the experiments are to be run under the root folder.   
- Main algorithms are implemented in LBPO.py and BACKTRACK.py.   

## Running Experiments

```
python LBPO.py --env <env_name> --exp_name <experiment name>     
python BACKTRACK.py --env <env_name> --exp_name <experiment name>     
```

## Environments
Safety environments:  Safexp-{robot}{task}{difficulty}-v0        
Choose robot from {Point, Car, Doggo}, task from {Goal, Push} and difficulty from {1,2}.   


## Important Note
Parts of the codes are used from the references mentioned below:
- spinning_up: https://github.com/openai/spinningup
- safety_starter_agents: https://github.com/openai/safety-starter-agents
- pytorch-a2c-ppo-acktr-gail: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail





