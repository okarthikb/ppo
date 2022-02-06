import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
from collections import deque
from time import sleep


class PPO:
    def __init__(self, env, actions, frames, phi, Actor, Critic, normalize, K, N, epochs, lr, gamma, eps):
        self.env = env
        self.actions = actions
        self.phi = phi
        self.frames = frames
        self.pi = Actor()
        self.pio = Actor()
        self.Q = Critic()
        self.K = K
        self.N = N
        self.epochs = epochs
        self.gamma = gamma
        self.eps = eps
        self.normalize = normalize
        self.opt = Adam(list(self.pi.parameters()) + list(self.Q.parameters()), lr)

    def play(self, render=True, fps=50):
        done = False
        G = 0
        s = self.env.reset()
        s = self.env.reset()
        seq = deque([np.zeros_like(s)] * (self.frames - 1) +
                    [s], maxlen=self.frames)
        with torch.no_grad():
            while not done:
                s = torch.tensor(self.phi(seq), dtype=torch.float32)
                probs = self.pio(s).squeeze()
                i = Categorical(probs).sample().item()
                s, r, done, _ = self.env.step(self.actions[i])
                G += r
                seq.append(s)
                if render:
                    self.env.render()
                    sleep(1 / fps)
        self.env.close()
        return G

    def accumulate(self, rs):
        Gs = [r for r in rs]
        for t in range(len(Gs) - 2, -1, -1):
            Gs[t] += self.gamma * Gs[t + 1]
        return torch.tensor(Gs, dtype=torch.float32)

    def rollout(self):
        done = False
        S, a, rs, oldps = [], [], [], []
        G = 0
        s = self.env.reset()
        seq = deque([np.zeros_like(s)] * (self.frames - 1) +
                    [s], maxlen=self.frames)
        with torch.no_grad():
            while not done:
                s = torch.tensor(self.phi(seq), dtype=torch.float32)
                S.append(s.squeeze(0))
                probs = self.pio(s).squeeze()
                i = Categorical(probs).sample().item()
                a.append(torch.tensor([i]))
                oldps.append(probs[i])
                s, r, done, _ = self.env.step(self.actions[i])
                G += r
                rs.append(r)
                seq.append(s)
        S, a, oldps = torch.stack(S), torch.stack(a), torch.stack(oldps)
        Gs = self.accumulate(rs)
        return (S, a, Gs, oldps), G

    def compute_episode_loss(self, S, a, Gs, oldps):
        Qs = torch.gather(self.Q(S), 1, a).squeeze()
        ps = torch.gather(self.pi(S), 1, a).squeeze()
        A = (Qs - Gs).detach()
        if self.normalize:
            A /= torch.linalg.norm(A)
        ratio = ps / oldps
        clipped_ratio = torch.clip(ratio, 1 - self.eps, 1 + self.eps)
        actor_loss = -torch.minimum(A * ratio, A * clipped_ratio).mean()
        critic_loss = F.mse_loss(Qs, Gs)
        return actor_loss + critic_loss

    def compute_avg_loss(self, rollouts):
        return sum(self.compute_episode_loss(*T) for T, _ in rollouts) / len(rollouts)

    def compute_avg_return(self, rollouts):
        return sum(G for _, G in rollouts) / len(rollouts)

    def train(self, log_interval):
        i = 0
        returns = []
        for k in range(1, self.K + 1):
            self.pio.load_state_dict(self.pi.state_dict())
            rollouts = [self.rollout() for _ in range(self.N)]
            returns.append(self.compute_avg_return(rollouts))
            for _ in range(self.epochs):
                avg_loss = self.compute_avg_loss(rollouts)
                self.opt.zero_grad()
                avg_loss.backward()
                self.opt.step()
            if k % log_interval == 0:
                i += 1
                torch.save(self.pio.state_dict(), f"checkpoints/pi{i}.pt")
                print("episode: {}\treturn: {:.2f}".format(
                    k * self.N, returns[-1]))
        return returns
