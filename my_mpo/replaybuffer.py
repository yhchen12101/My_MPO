import numpy as np

class ReplayBuffer:
    def __init__(self):
        #buffers
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.buffer = []
        self.length = 0
        
    def clear(self):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.buffer = []
        self.length = 0
            
            
    def store(self, episodes):
        for episode in episodes:
            states, actions, next_states, rewards = zip(*episode)
            
            # need to modified for multi-worker
            
            episode_len = len(states)
            usable_episode_len = episode_len - 1
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
            self.idx_to_episode_idx.extend([len(self.buffer)] * usable_episode_len)
            self.buffer.append((states, actions, next_states, rewards))
            """
            self.buffer.append((states, actions, next_states, rewards))
            self.length = len(states)
            """
    """
    def __getitem__(self, idx):
        #need to modified for multi-worker
        state, action, next_state, reward = self.buffer[0]
        return state[idx], action[idx], next_state[idx], reward[idx]
    """
    def __getitem__(self, idx):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx
        states, actions, next_states, rewards = self.buffer[episode_idx]
        state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
        return state, action, next_state, reward
    
    def __len__(self):
        return len(self.idx_to_episode_idx)
        #return self.length
        
    def reward_mean(self):
        _, _, _, rewards = zip(*self.buffer)
        return np.mean([np.mean(reward) for reward in rewards])
