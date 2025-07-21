import torch
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
        
        self.env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

    def run(self, is_training=True, render=False):
        
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            epsilon_decay = self.epsilon_decay
            
            
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
                
                # Move to new state
                state = new_state
                
            rewards_per_episode.append(episode_reward)
            
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)
                
            print(f"Episode {episode+1}: Reward = {episode_reward}")
                
                

        self.env.close()

if __name__ == '__main__':
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)
