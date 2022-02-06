from ppo import *


env = gym.make("CartPole-v0")
actions = [0, 1]
frames = 1


def phi(seq):
  return np.array(seq)


class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc = nn.Linear(4, 64)
    self.hidden = nn.Linear(64, 64)
    self.out = nn.Linear(64, len(actions))

  def forward(self, x):
    x = F.relu(self.fc(x))
    x = F.relu(self.hidden(x))
    x = F.softmax(self.out(x), -1)
    return x


class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc = nn.Linear(4, 64)
    self.hidden = nn.Linear(64, 64)
    self.out = nn.Linear(64, len(actions))

  def forward(self, x):
    x = F.relu(self.fc(x))
    x = F.relu(self.hidden(x))
    x = self.out(x)
    return x


normalize = True
K = 200
N = 32
epochs = 128
lr = 3e-3
gamma = 0.999
eps = 0.2
log_interval = 10


agent = PPO(env, actions, frames, phi, Actor, Critic, normalize, K, N, epochs, lr, gamma, eps)

if __name__ == "__main__":
    returns = agent.train(log_interval)
    fig = plt.figure(figsize=(8, 8))
    plt.xlabel("episode")
    plt.ylabel("return")
    plot = plt.plot(N * np.arange(1, K + 1), returns)
    plt.savefig("checkpoints/returns.png")
