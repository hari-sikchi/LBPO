import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import copy


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPDeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,discount_factor=0.99):
        super(MLPDeterministicActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)


        self.max_action = max_action
        self.action_dim=action_dim
        self.start_state = None
        


    def forward(self, state,safety_switch=False,debug = False, noisy=False):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def mean(self, obs):
        mu = self.mu_net(obs)
        return mu

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCriticTD3trust(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # policy builder depends on action space
        self.pi = MLPDeterministicActor(obs_dim, action_space.shape[0],action_space.high[0])


        # build value function
        self.Qv1  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)
        self.Qv2  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)


        self.Qj1  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)
        self.Qj2  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)

        self.baseline_Qj = copy.deepcopy(self.Qj1)
        self.baseline_pi = copy.deepcopy(self.pi)
        self.pi_mix = copy.deepcopy(self.pi)
        self.epsilon = 0


    def act_with_correction(self,pi,obs,max_step_size):
        # import ipdb; ipdb.set_trace()
        act = pi(obs)
        mixing_parameter = 0.0
        baseline_action = self.baseline_pi(obs)

        # Zero out all previous gradients
        self.baseline_pi.zero_grad()
        self.baseline_Qj.zero_grad()

        # Find the gradient of cost critic with respect to action on baseline action
        obs_act = torch.cat((obs, baseline_action),dim=1)
        obs_act = obs_act.requires_grad_(True)
        obs_act.retain_grad()
        cost = self.baseline_Qj(obs_act).sum()
        cost.backward(retain_graph=True)
        
        grad_baseline_action = obs_act.grad[:,-self.act_dim:].detach() 

        baseline_action = baseline_action.detach()

        lambda_star = F.relu(((1 - mixing_parameter) * torch.sum(grad_baseline_action * (act-baseline_action),dim=1).view(-1,1) - self.epsilon)\
                        /(torch.sum(grad_baseline_action*grad_baseline_action,dim=1).view(-1,1)+1e-6))


        # lambda_star = lambda_star/lambda_star.norm()
        # lr = 0.01
        update = torch.clamp(lambda_star*grad_baseline_action,-max_step_size,max_step_size)
        # update = torch.min(lambda_star*grad_baseline_action, torch.Tensor([max_step_size]))
        # update = update/ (update.norm(dim=1).view(-1,1)+1e-6)

        corrected_action = act - update

        return corrected_action     

    def step(self, obs):
        a = self.pi(obs)
        qv = self.Qv1(torch.cat((obs,a)))
        qj = self.Qj1(torch.cat((obs,a)))
        return a.detach().cpu().numpy(), qv.detach().cpu().numpy(), qj.detach().cpu().numpy(), 0


    def act_pi(self, pi, obs):
        a = pi(obs)
        return a

    def act(self, obs):
        return self.step(obs)[0]


class MLPActorCriticTD3trustCQL(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # policy builder depends on action space
        self.pi = MLPDeterministicActor(obs_dim, action_space.shape[0],action_space.high[0])


        # build value function
        self.Qv1  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)
        self.Qv2  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)


        self.Qj1  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)
        self.Qj2  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)


        self.cql_Qj1  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)
        self.cql_Qj2  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)

        self.baseline_cql_Qj = copy.deepcopy(self.cql_Qj1)
        self.baseline_Qj = copy.deepcopy(self.Qj1)
        self.baseline_pi = copy.deepcopy(self.pi)
        self.pi_mix = copy.deepcopy(self.pi)
        self.epsilon = 0


    def act_with_correction(self,pi,obs,max_step_size):
        # import ipdb; ipdb.set_trace()
        act = pi(obs)
        mixing_parameter = 0.0
        baseline_action = self.baseline_pi(obs)

        # Zero out all previous gradients
        self.baseline_pi.zero_grad()
        self.baseline_Qj.zero_grad()
        self.baseline_cql_Qj.zero_grad()

        # Find the gradient of cost critic with respect to action on baseline action
        obs_act = torch.cat((obs, baseline_action),dim=1)
        obs_act = obs_act.requires_grad_(True)
        obs_act.retain_grad()

        diff = -self.baseline_cql_Qj(obs_act) - self.baseline_Qj(obs_act)

        epsilon_ = torch.FloatTensor(self.epsilon - diff.cpu().detach().numpy())

        cost = diff.sum()
        cost.backward(retain_graph=True)
        
        grad_baseline_action = obs_act.grad[:,-self.act_dim:].detach() 

        baseline_action = baseline_action.detach()

        lambda_star = F.relu(((1 - mixing_parameter) * torch.sum(grad_baseline_action * (act-baseline_action),dim=1).view(-1,1) - epsilon_.view(-1,1))\
                        /(torch.sum(grad_baseline_action*grad_baseline_action,dim=1).view(-1,1)+1e-6))


        # lambda_star = lambda_star/lambda_star.norm()
        # lr = 0.01
        update = torch.clamp(lambda_star*grad_baseline_action,-max_step_size,max_step_size)
        # update = torch.min(lambda_star*grad_baseline_action, torch.Tensor([max_step_size]))
        # update = update/ (update.norm(dim=1).view(-1,1)+1e-6)

        corrected_action = act - update

        return corrected_action     

    def step(self, obs):
        a = self.pi(obs)
        qv = self.Qv1(torch.cat((obs,a)))
        qj = self.Qj1(torch.cat((obs,a)))
        cql_qj = self.cql_Qj1(torch.cat((obs,a)))
        return a.detach().cpu().numpy(), qv.detach().cpu().numpy(), qj.detach().cpu().numpy(),cql_qj.detach().cpu().numpy(), 0


    def act(self, obs):
        return self.step(obs)[0]


class MLPActorCriticTD3(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # policy builder depends on action space
        self.pi = MLPDeterministicActor(obs_dim, action_space.shape[0],action_space.high[0])


        # build value function
        self.Qv  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)
        # self.j  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.Qj  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)

        self.baseline_Qj = copy.deepcopy(self.Qj)
        self.baseline_pi = copy.deepcopy(self.pi)
        self.pi_mix = copy.deepcopy(self.pi)
        self.epsilon = 0


    def act_with_correction(self,pi,obs,max_step_size):
        # import ipdb; ipdb.set_trace()
        act = pi(obs)
        mixing_parameter = 0.0
        baseline_action = self.baseline_pi(obs)

        # Zero out all previous gradients
        self.baseline_pi.zero_grad()
        self.baseline_Qj.zero_grad()

        # Find the gradient of cost critic with respect to action on baseline action
        obs_act = torch.cat((obs, baseline_action),dim=1)
        obs_act = obs_act.requires_grad_(True)
        obs_act.retain_grad()
        cost = self.baseline_Qj(obs_act).sum()
        cost.backward(retain_graph=True)
        
        grad_baseline_action = obs_act.grad[:,-self.act_dim:].detach() 

        baseline_action = baseline_action.detach()

        lambda_star = F.relu(((1 - mixing_parameter) * torch.sum(grad_baseline_action * (act-baseline_action),dim=1).view(-1,1) - self.epsilon)\
                        /(torch.sum(grad_baseline_action*grad_baseline_action,dim=1).view(-1,1)+1e-6))


        # lambda_star = lambda_star/lambda_star.norm()
        # lr = 0.01
        update = torch.min(lambda_star*grad_baseline_action, torch.Tensor([max_step_size]))
        # update = update/ (update.norm(dim=1).view(-1,1)+1e-6)

        corrected_action = act - update

        return corrected_action     

    def step(self, obs):
        a = self.pi(obs)
        qv = self.Qv(torch.cat((obs,a)))
        qj = self.Qj(torch.cat((obs,a)))
        return a.detach().cpu().numpy(), qv.detach().cpu().numpy(), qj.detach().cpu().numpy(), 0


    def act(self, obs):
        return self.step(obs)[0]

        
class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class MLPActorCriticLyapunov(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.pi_mix = copy.deepcopy(self.pi)
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        # self.j  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.Qj  = MLPCritic(obs_dim+act_dim, hidden_sizes, activation)

        self.baseline_Qj = copy.deepcopy(self.Qj)
        self.baseline_pi = copy.deepcopy(self.pi)
        self.epsilon = 0


    def act_with_correction(self, pi, obs, max_step_size=0.05):
        act = pi.mean(obs)
        mixing_parameter = 0.0
        baseline_action = self.baseline_pi.mean(obs)

        # Zero out all previous gradients
        self.baseline_pi.zero_grad()
        self.baseline_Qj.zero_grad()

        # Find the gradient of cost critic with respect to action on baseline action
        obs_act = torch.cat((obs, baseline_action),dim=1)
        obs_act = obs_act.requires_grad_(True)
        obs_act.retain_grad()
        cost = self.baseline_Qj(obs_act).sum()
        cost.backward(retain_graph=True)
        
        grad_baseline_action = obs_act.grad[:,-self.act_dim:].detach() 

        baseline_action = baseline_action.detach()

        lambda_star = F.relu(((1 - mixing_parameter) * torch.sum(grad_baseline_action * (act-baseline_action),dim=1).view(-1,1) - self.epsilon)\
                        /(torch.sum(grad_baseline_action*grad_baseline_action,dim=1).view(-1,1)+1e-6))


        # lambda_star = lambda_star/lambda_star.norm()
        # lr = 0.01
        update = torch.min(lambda_star*grad_baseline_action, torch.Tensor([max_step_size]))
        # update = update/ (update.norm(dim=1).view(-1,1)+1e-6)

        corrected_action = act - update

        return corrected_action     
        

    def get_action_grad(self, obs):
        pi = self.pi._distribution(obs)
        act = pi.rsample()
        mixing_parameter = 0.0
        baseline_pi = self.baseline_pi._distribution(obs)
        baseline_action = baseline_pi.sample()

        # Zero out all previous gradients
        self.baseline_pi.zero_grad()
        self.baseline_Qj.zero_grad()

        # Find the gradient of cost critic with respect to action on baseline action
        # import ipdb; ipdb.set_trace()
        obs_act = torch.cat((obs, baseline_action),dim=1)
        obs_act = obs_act.requires_grad_(True)
        cost = self.baseline_Qj(obs_act).sum()
        cost.backward(retain_graph=True)
        
        grad_baseline_action = obs_act.grad[:,-self.act_dim:].detach() 

        baseline_action = baseline_action.detach()
        # import ipdb; ipdb.set_trace()

        lambda_star = F.relu(((1 - mixing_parameter) * torch.sum(grad_baseline_action * (act-baseline_action),dim=1).view(-1,1) - self.epsilon)\
                        /(torch.sum(grad_baseline_action*grad_baseline_action,dim=1)).view(-1,1))


        # lambda_star = lambda_star/lambda_star.norm()
        lr = 0.01
        update = lambda_star*grad_baseline_action
        update = update/ (update.norm(dim=1).view(-1,1)+1e-6)

        corrected_action = act - lr*update
        # pi = self.pi._distribution(obs)
        logp = self.pi._log_prob_from_distribution(pi, act)

        return corrected_action, logp




    def apply_correction(self, obs, act):


        mixing_parameter = 0.0
        baseline_pi = self.baseline_pi._distribution(obs)
        baseline_action = baseline_pi.sample()

        # Zero out all previous gradients
        self.baseline_pi.zero_grad()
        self.baseline_Qj.zero_grad()

        obs_act = torch.cat((obs, baseline_action))
        obs_act.requires_grad_(True)

        # Find the gradient of cost critic with respect to action on baseline action
        cost = self.baseline_Qj(obs_act).sum()
        cost.backward(retain_graph=True)
        # import ipdb; ipdb.set_trace()
        grad_baseline_action = obs_act.grad.detach()[-self.act_dim:] 
        
        baseline_action = baseline_action.detach().cpu().numpy()
        a_np = act.detach().cpu().numpy()

        
        lambda_star = F.relu((torch.sum(grad_baseline_action.view(1,-1) * (a_np-baseline_action).reshape(1,-1),dim=1).view(1,-1) - self.epsilon)/(torch.sum(grad_baseline_action.view(1,-1)*grad_baseline_action.view(1,-1),dim=1)).view(1,-1))


        # lambda_star = F.relu(((1 - mixing_parameter) * torch.sum(grad_baseline_action.view(1,-1) * (a_np-baseline_action).reshape(1,-1),dim=1).view(1,-1) - self.epsilon)\
        #                 /(torch.sum(grad_baseline_action.view(-1,1)*grad_baseline_action.view(-1,1),dim=1)).view(-1,1))


        lr = 0.05
        update = lambda_star*grad_baseline_action
        update = update/ (update.norm(dim=1).view(-1,1)+1e-6)
        corrected_action = act - lr*update
        # corrected_action = a_np - lambda_star*grad_baseline_action
        pi = self.pi._distribution(obs)
        logp = self.pi._log_prob_from_distribution(pi, corrected_action)
        # import ipdb; ipdb.set_trace()
        return corrected_action.view(-1), logp



    def step(self, obs):
        
        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        # j = self.j(obs)
        qj = self.Qj(torch.cat((obs,a)))
        
        
        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), qj.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

        


    def act(self, obs):
        return self.step(obs)[0]


class MLPActorCriticCost(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.j  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            j = self.j(obs)
        return a.numpy(), v.numpy(), j.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

