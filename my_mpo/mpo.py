import torch
import torch.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import MultivariateNormal
from scipy.optimize import minimize
from torch.nn.utils import clip_grad_norm_

from my_mpo.policy import Policy_Net
from my_mpo.q_function import Q_Net
import my_mpo.utils as utils
from my_mpo.replaybuffer import ReplayBuffer

class MPO(object):
    def __init__(self, args, env):
        if args.device == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(args.device))
        self.save_name = args.save_name
        # environment related
        #self.env = args.env
        self.env = env
        self.domain = args.domain
        self.task = args.task
        self.continuous_action_sapce = args.c
        
        #training related
        self.policy_hidden = args.policy_hidden   # hidden layer dimension of policy network
        self.q_hidden = args.q_hidden   #hidden layer dimension of Q function network
        self.discount = args.discount  # discount factor
        self.lr = args.lr
        
        # about largrangian multiplier
        self.eta = np.random.rand() #lagrangian multiplier for continuous action space in the E-step
        self.eta_mu = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.eta_cov = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.eta_epsilon = args.epsilon   # hard constraint for constrained E-step
        self.eta_mu_epsilon = args.epsilon_mu   # hard constraint of mu for variational distribution
        self.eta_cov_epsilon =args.epsilon_cov  # hard constraint of covariance for variational distribution
        
        # evaluation
        self.evaluate_episode_num = 30
        self.evaluate_episode_maxstep = 300
        self.evaluate_period = 1
        self.model_save_period = 10
        self.max_return_eval = -np.inf
        
        # need to modify
        self.eta_mu_scale = 1
        self.eta_cov_scale = 100
        self.worker_num = 50  # number of parallel sample worker
        self.step_max = 300 # maximum number of step in each episode
        self.sample_action_num = 64 # number of sample action of each state for policy evaluation
        self.batch_size = 128 
        self.loss_q = nn.MSELoss()  #nn.SmoothL1Loss()
        self.mstep_iteration_num = 5 # iter num of m step update
        
        self.start_iteration = 1
        
        #generate policy network and Q-function network
        if self.continuous_action_sapce:
            self.policy = Policy_Net(self.env, self.policy_hidden).to(self.device)
            self.target_policy = Policy_Net(self.env, self.policy_hidden).to(self.device)
            
            self.q_function = Q_Net(self.env, self.q_hidden).to(self.device)
            self.target_q_function = Q_Net(self.env, self.q_hidden).to(self.device)
       
        self.target_policy = utils.copy_model(self.policy, self.target_policy)
        self.target_q_policy = utils.copy_model(self.q_function, self.target_q_function)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q_function_optimizer = torch.optim.Adam(self.q_function.parameters(), lr=self.lr)

        self.replaybuffer = ReplayBuffer()
        
    def train(self, iters=1000):
        log, writer = utils.set_save_path(self.save_name)
        timer = utils.Timer()
        
        for iter in range(self.start_iteration, iters):
            log_info = ['epoch {}/{}'.format(iter, iters)]
            t_epoch_start = timer.t()
            Evaluation_loss = 0
            Improve_loss = 0
                        
            self.sample_trajectory()
  
            for r in range(3):
                for indices in tqdm(BatchSampler(SubsetRandomSampler(range(len(self.replaybuffer))), self.batch_size, drop_last=True), leave=False, desc='training {}/{}'.format(r+1, 3)):
                    state_length = len(indices)
                    batch_state, batch_action, batch_next_state, batch_reward = zip(*[self.replaybuffer[index] for index in indices])
                    
                    batch_state = torch.from_numpy(np.stack(batch_state)).type(torch.float32).to(self.device)
                    batch_action = torch.from_numpy(np.stack(batch_action)).type(torch.float32).to(self.device)
                    batch_next_state = torch.from_numpy(np.stack(batch_next_state)).type(torch.float32).to(self.device) 
                    batch_reward = torch.from_numpy(np.stack(batch_reward)).type(torch.float32).to(self.device)
                    
                    #Policy Evaluation
                    Evaluation_loss += self.Policy_Evaluation(batch_state, batch_action, batch_next_state, batch_reward)
                    
                    # Policy Improvement E-Step
                    mu_i, cholesky_i, sample_actions_i, q_i = self.E_step(batch_state)
                    
                    # Policy Improvement M-Step
                    Improve_loss += self.M_step(batch_state, mu_i, cholesky_i, sample_actions_i, q_i, state_length)
                                        
            Evaluation_loss/=(1*len(self.replaybuffer))
            Improve_loss/=(1*len(self.replaybuffer))
            log_info.append('eval loss={:.4f}, imporve loss={:.4f}'.format(Evaluation_loss, Improve_loss))
            writer.add_scalars('loss', {'Evaluation': Evaluation_loss}, iter) 
            writer.add_scalars('loss', {'Improve': Improve_loss}, iter) 
  
            t = timer.t()
            prog = iter / iters
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

            self.target_policy = utils.copy_model(self.policy, self.target_policy)
            self.target_q_function = utils.copy_model(self.q_function, self.target_q_function)
            
            return_eval = None
            if iter % self.evaluate_period == 0:
                self.policy.eval()
                return_eval = self.Evaluate()
                self.policy.train()
                self.max_return_eval = max(self.max_return_eval, return_eval)
                log_info.append('eval return={:.4f}, max return={:.4f}'.format(return_eval, self.max_return_eval))
                writer.add_scalars('return', {'eval return': return_eval}, iter) 
                writer.add_scalars('return', {'max eval return': self.max_return_eval}, iter) 
                
                self.save_model(os.path.join(utils._log_path, 'model_latest.pt'))
            if iter % self.model_save_period == 0:
                self.save_model(os.path.join(utils._log_path, 'model_{}.pt'.format(iter)))

            log_info.append('{}/{}'.format(t_elapsed, t_all))
            log(', '.join(log_info))
    
    def Policy_Evaluation(self, states, actions, next_states, rewards):
        # not implement discrete version yet
        with torch.no_grad():
            _, _, _, expected_next_q = self.Expect_q(next_states)
            expected_next_q = expected_next_q.mean(dim=1)
            target_q = rewards + self.discount * expected_next_q
        
        if self.continuous_action_sapce:
            # Q network
            self.q_function_optimizer.zero_grad()
            q = self.q_function(states, actions).squeeze()
            loss = self.loss_q(target_q, q)
            loss.backward()
            self.q_function_optimizer.step()
            return loss.item()
    
    def E_step(self, states):
        mu, cholesky, sample_actions, target_q = self.Expect_q(states)
        target_q = target_q.transpose(0, 1)
        sample_actions = sample_actions.transpose(0, 1)
        target_q_np = target_q.cpu().transpose(0, 1).numpy()
        # define dual function
        if self.continuous_action_sapce:
            def dual(eta):
                max_q = np.max(target_q_np, 1)
                return eta * self.eta_epsilon + np.mean(max_q) \
                        + eta * np.mean(np.log(np.mean(np.exp((target_q_np - max_q[:, None]) / eta), axis=1))) 
        bounds = [(1e-6, None)]
        
        res = minimize(dual, np.array([self.eta]), method='SLSQP', bounds=bounds)
        self.eta = res.x[0]
        qij = torch.softmax(target_q / self.eta, dim=0)  # (action dim, batch)
        return mu, cholesky, sample_actions, qij

    
    def M_step(self, states, mu_i, cholesky_i, sample_actions_i, q_i, state_length):
        for _ in range(self.mstep_iteration_num):
            if self.continuous_action_sapce:
                mu, cholesky = self.policy(states)
                action_dist_1 = MultivariateNormal(loc=mu_i, scale_tril=cholesky)
                action_dist_2 = MultivariateNormal(loc=mu, scale_tril=cholesky_i)
                loss_1 = torch.mean(q_i * (
                                    action_dist_1.expand((self.sample_action_num, state_length)).log_prob(sample_actions_i)
                                    + action_dist_2.expand((self.sample_action_num, state_length)).log_prob(sample_actions_i)))
            
                C_mu, C_cov = utils.gaussian_kl(
                    mu_i, mu,
                    cholesky_i, cholesky)   
                
                #update lagrange multipliers
                self.eta_mu -= self.eta_mu_scale * (self.eta_mu_epsilon - C_mu).detach().item()
                self.eta_cov -= self.eta_cov_scale * (self.eta_cov_epsilon- C_cov).detach().item()
                
                self.eta_mu = np.clip(0.0, self.eta_mu, 0.1)
                self.eta_cov = np.clip(0.0, self.eta_cov, 10.0)
                self.policy_optimizer.zero_grad()
                loss = -(loss_1
                         + self.eta_mu * (self.eta_mu_epsilon - C_mu)
                         + self.eta_cov * (self.eta_cov_epsilon - C_cov))
                loss.backward()
                clip_grad_norm_(self.policy.parameters(), 0.1)
                self.policy_optimizer.step()                
        return loss.item()
                
    def Expect_q(self, states):
        with torch.no_grad():
            if self.continuous_action_sapce:
                # target Q network
                mu, cholesky = self.target_policy(states)
                action_distribution = MultivariateNormal(mu, scale_tril=cholesky)
                sample_actions = action_distribution.sample((self.sample_action_num,)).transpose(0, 1)
                expanded_states = states[:, None, :].expand(-1, self.sample_action_num, -1)
                expected_q = self.target_q_function(
                    expanded_states.reshape(-1, expanded_states.size(-1)),  # (batch * sample_num, state_dim)
                    sample_actions.reshape(-1,sample_actions.size(-1))  # (batch * sample_num, action_dim)
                ).reshape(self.batch_size, -1)  # (batch * sample_num)
        return mu, cholesky, sample_actions, expected_q
                
    def sample_trajectory(self):
        self.replaybuffer.clear()
        episode_trajectory = [self.worker_trajectory() for i in tqdm(range(self.worker_num), leave=False, desc='trajectory_sampling')]
        self.replaybuffer.store(episode_trajectory)
        
    def worker_trajectory(self):
        buffer = []
        
        ############for gym###################
        state = self.env.reset()
        
        for steps in range(self.step_max):
            action = self.target_policy.action(
                torch.from_numpy(state).type(torch.float32).to(self.device)
            ).cpu().numpy()
            next_state, reward, done, _ = self.env.step(action)
            buffer.append((state, action, next_state, reward))
            if done:
                break
            else:
                state = next_state
        ######################################  
        """
        ############for suite###################
        time_step_0 = self.env.reset()
        state = time_step_0.observation['observations']
        for steps in range(self.step_max):
            action = self.target_policy.action(
                torch.from_numpy(state).type(torch.float32).to(self.device)
            ).cpu().numpy()
            next_state, reward, done, _ = utils.separate_info(self.env.step(action))
            buffer.append((state, action, next_state, reward))
            if done:
                break
            else:
                state = next_state
        ###################################### 
        """                
        return buffer
        
    def Evaluate(self):
        """
        :return: average return over 100 consecutive episodes
        """
        with torch.no_grad():
            total_rewards = []
            for e in tqdm(range(self.evaluate_episode_num), leave=False, desc='evaluating'):
                total_reward = 0.0
                
                ############for gym###################
                state = self.env.reset()
                ######################################  
                """
                ############for suite###################
                time_step_0 = self.env.reset()
                state = time_step_0.observation['observations']  
                ######################################  
                """
                for s in range(self.evaluate_episode_maxstep):
                    action = self.policy.action(
                        torch.from_numpy(state).type(torch.float32).to(self.device)
                    ).cpu().numpy()
                    
                    ############for gym###################
                    state, reward, done, _ = self.env.step(action)
                    ######################################  
                    """
                    ############for suite###################   
                    state, reward, done, _ =  utils.separate_info(self.env.step(action))
                    ######################################  
                    """
                    total_reward += reward
                    if done:
                        break
                total_rewards.append(total_reward)
            return np.mean(total_rewards)
        
    def save_model(self, path=None):
        """
        saves a model to a given path
        :param path: (str) file path (.pt file)
        """
        data = {
            'actor_state_dict': self.policy.state_dict(),
            'target_actor_state_dict': self.target_policy.state_dict(),
            'critic_state_dict': self.q_function.state_dict(),
            'target_critic_state_dict': self.target_q_function.state_dict(),
            'actor_optim_state_dict': self.policy_optimizer.state_dict(),
            'critic_optim_state_dict': self.q_function_optimizer.state_dict()
        }
        torch.save(data, path)
        