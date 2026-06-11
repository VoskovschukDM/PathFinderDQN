"""Базовый DQN-агент для проекта PathFinder DQN."""

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """Нейронная сеть, которая оценивает Q-значения для каждого действия."""

    def __init__(self, obs_n, num_actions, embed_dim=128):
        super().__init__()

        # Состояние задаётся целым индексом клетки, поэтому сначала переводим его в embedding-вектор.
        self.embedding = nn.Embedding(obs_n, embed_dim)

        # Полносвязная часть сети выдаёт Q-value для каждого возможного действия.
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state_idx):
        """Прямой проход сети: индекс состояния -> набор Q-значений."""
        x = self.embedding(state_idx.long())
        return self.fc(x)


class DQNAgent:
    """Агент Deep Q-Network с replay buffer и epsilon-greedy стратегией."""

    def __init__(self, env, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10000):
        self.env = env
        self.gamma = gamma  # Коэффициент дисконтирования будущей награды.
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)  # Память переходов: (state, action, reward, next_state, done).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(env.obs_n, env.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state_idx, eps=0.1):
        """Выбирает действие: случайное с вероятностью eps или лучшее по Q-value."""
        if random.random() < eps:
            return random.randint(0, self.env.num_actions - 1)

        state_tensor = torch.tensor([state_idx], device=self.device)
        q_vals = self.model(state_tensor)
        return q_vals.argmax().item()

    def remember(self, s, a, r, s_, done):
        """Сохраняет один переход в replay buffer."""
        self.buffer.append((s, a, r, s_, done))

    def replay(self):
        """Обучает модель на случайной мини-выборке из накопленного опыта."""
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, done = zip(*batch)

        # Переводим данные из Python-структур в tensor-формат PyTorch.
        s = torch.tensor(s, device=self.device)
        s_ = torch.tensor(s_, device=self.device)
        a = torch.tensor(a, device=self.device).unsqueeze(1)
        r = torch.tensor(r, device=self.device).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) — оценка текущей модели для реально выбранных действий.
        q_vals = self.model(s).gather(1, a)

        # Target по формуле Беллмана: reward + gamma * max Q(next_state).
        q_next = self.model(s_).max(1)[0].unsqueeze(1).detach()
        q_target = r + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="dqn_model.pth"):
        """Сохраняет веса модели в файл."""
        torch.save(self.model.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load(self, path="dqn_model.pth"):
        """Загружает веса модели из файла."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Модель загружена из {path}")

    def train(self, episodes=500, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        """Запускает обучение агента в одной среде и возвращает историю наград/успехов."""
        rewards, passes = [], []
        eps = eps_start

        for ep in range(episodes):
            state_idx, _ = self.env.reset()
            total_reward, done, steps = 0, False, 0

            while not done and steps < self.env.max_steps:
                action = self.act(state_idx, eps)
                next_state_idx, reward, terminated, truncated, info = self.env.step(action)
                self.remember(state_idx, action, reward, next_state_idx, terminated)
                self.replay()

                state_idx = next_state_idx
                total_reward += reward
                done = terminated or truncated
                steps += 1

            eps = max(eps_end, eps * eps_decay)
            rewards.append(total_reward)
            passes.append(1 if terminated else 0)

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}: reward={total_reward:.2f}, pass={passes[-1]}, eps={eps:.3f}")

        return rewards, passes
