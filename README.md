# Lyapunov Barrier Policy Optimization

This code reproduces the results for our method and baselines showed in the paper.[[ArXiv]](https://arxiv.org/abs/2103.09230).

If you use this code in your research project please cite us as:
```
@article{sikchi2021lyapunov,
  title={Lyapunov barrier policy optimization},
  author={Sikchi, Harshit and Zhou, Wenxuan and Held, David},
  journal={arXiv preprint arXiv:2103.09230},
  year={2021}
}
{"mode":"full","isActive":false}
```



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





