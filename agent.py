import torch
from torch import nn
import random
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from exp_replay import ReplayMemory
import itertools
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set, render = False):
        with open('hyper_param.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)
        
        self.render = render
        self.replay_memory_size = hyperparameters['replay_memory_size'] # size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size'] # size of the training data set samples
        self.epsilon_init = hyperparameters['epsilon_init'] # 1 = 100% random actions
        self.epsilon_decay = hyperparameters['epsilon_decay'] # epsilon decay rate
        self.epsilon_min = hyperparameters['epsilon_min'] # minimum epsilon value
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.loss_fn = nn.MSELose()
        self.optimizer = None

    def run(self, is_training=True, render=False):
        
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            step_count=0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            
            
        for episode in itertools.count():
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            terminated = False
            episode_reward = 0.0
            
            while not terminated:
                # Next action
                if is_training and random.random() < epsilon:
                    action = self.env.action_space.sample()  # chọn hành động ngẫu nhiên
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax() # Tim chi so co phan tu lon nhat. Bat dau tu do
                
                # Processing:
                new_state, reward, terminated, _, info = self.env.step(action.item())
                
                # Accumulate reward    
                episode_reward += reward
                
                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                   
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    
                    # Increment step counter
                    step_count += 1
                
                # Move to new state
                state = new_state
                
            rewards_per_episode.append(episode_reward)
            
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)
            
            if len(memory) > self.mini_batch_size:
               mini_batch = memory.sample(self.mini_batch_size)
               self.optimize(mini_batch, policy_dqn, target_dqn)
               
               # Copy policy network to target network after a certain number of steps
               if step_count > self.network_sync_rate:
                   target_dqn.load_state_dict(policy_dqn.dict())
                   step_count = 0
                
            print(f"Episode {episode+1}: Reward = {episode_reward}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
                    
            current_q = policy_dqn(state)
            
            # Compute the loss for each mini batch
            loss = self.loss_fn(current_q, target_q)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)
