import flappy_bird_gymnasium
import gymnasium
from dqn import DQN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, env_name="CartPole-v1", render_mode="human"):
        self.env = gymnasium.make(env_name, render_mode=render_mode)

    def run(self, is_training=True, render=False):
        
        done = False
        
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions)
        
        obs, _ = self.env.reset()
        while not done:
            action = self.env.action_space.sample()  # chọn hành động ngẫu nhiên
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Dừng nếu kết thúc episode
            if terminated or truncated:
                done = True

        self.env.close()

# Sử dụng Agent
agent = Agent()
agent.run()
