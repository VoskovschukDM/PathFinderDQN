"""PathFinder DQN: запуск обучения и проверки агента на наборе карт.

Скрипт делит карты на обучающую и валидационную выборки, обучает DQN-агента,
строит графики качества и сохраняет веса обученной модели.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import DQNAgent  # При необходимости можно заменить на DoubleDQNAgent / DuelingDQNAgent.
from enviroment import make_env


PROJECT_NAME = "PathFinder DQN"


def moving_average(x, w):
    """Возвращает скользящее среднее по окну w для сглаживания графиков."""
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_map(file_path):
    """Загружает текстовую карту и преобразует её в список символов."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [list(line.strip()) for line in f if line.strip()]


if __name__ == "__main__":
    print(f"=== {PROJECT_NAME} ===")

    # Ищем карты верхнего уровня в папке maps. Каждая карта — это txt-файл с S, G и H.
    maps_dir = Path("maps")
    map_files = sorted(maps_dir.glob("*.txt"))
    if not map_files:
        raise FileNotFoundError("Нет карт в папке 'maps'")

    # Делим карты на train и validation, чтобы оценивать модель на неиспользованных при обучении данных.
    random.seed(42)
    train_size = int(0.7 * len(map_files))
    train_files = random.sample(map_files, train_size)
    val_files = [f for f in map_files if f not in train_files]

    print(f"Training on {len(train_files)} maps, validation on {len(val_files)} maps")

    # Создаём агента. Первая среда нужна только для определения размеров пространства состояний и действий.
    agent = DQNAgent(make_env(load_map(train_files[0])))
    episodes_per_map = 50
    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995

    # ======= TRAINING =======
    rewards_train, passes_train = [], []
    eps = eps_start

    for ep in range(episodes_per_map):
        # На каждом эпизоде выбираем случайную карту, чтобы агент учился на разных конфигурациях.
        map_file = random.choice(train_files)
        desc = load_map(map_file)
        env = make_env(desc)
        state_idx, _ = env.reset()

        total_reward, done, steps = 0, False, 0
        while not done and steps < env.max_steps:
            # Epsilon-greedy: часть действий случайная, часть выбирается моделью.
            action = agent.act(state_idx, eps)
            next_state_idx, reward, terminated, truncated, info = env.step(action)

            # Сохраняем переход в replay buffer и делаем один шаг обучения.
            agent.remember(state_idx, action, reward, next_state_idx, terminated)
            agent.replay()

            state_idx = next_state_idx
            total_reward += reward
            done = terminated or truncated
            steps += 1

        # Постепенно уменьшаем вероятность случайных действий.
        eps = max(eps_end, eps * eps_decay)
        rewards_train.append(total_reward)
        passes_train.append(1 if terminated else 0)

        if (ep + 1) % 10 == 0:
            print(f"[Train] Episode {ep + 1}: reward={total_reward:.2f}, pass={passes_train[-1]}, eps={eps:.3f}")

    # ======= VALIDATION =======
    rewards_val, passes_val = [], []
    for map_file in val_files:
        desc = load_map(map_file)
        env = make_env(desc)
        state_idx, _ = env.reset()

        total_reward, done, steps = 0, False, 0
        while not done and steps < env.max_steps:
            # На валидации eps=0: проверяем только стратегию, выученную моделью.
            action = agent.act(state_idx, eps=0.0)
            next_state_idx, reward, terminated, truncated, info = env.step(action)
            state_idx = next_state_idx
            total_reward += reward
            done = terminated or truncated
            steps += 1

        rewards_val.append(total_reward)
        passes_val.append(1 if terminated else 0)
        print(f"[Val] Map {map_file.name}: reward={total_reward:.2f}, pass={1 if terminated else 0}")

    # ======= Графики =======
    # Слева отображается награда, справа — факт успешного прохождения карты.
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_train, label="Train Reward")
    plt.plot(range(len(rewards_train) - 49, len(rewards_train)), moving_average(rewards_train, 50), label="MA50 Train")
    plt.plot(range(len(val_files)), rewards_val, "o-", label="Validation Reward")
    plt.xlabel("Episode / Map")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(passes_train, label="Train Pass")
    plt.plot(range(len(passes_train) - 49, len(passes_train)), moving_average(passes_train, 50), label="MA50 Train Pass")
    plt.plot(range(len(val_files)), passes_val, "o-", label="Validation Pass")
    plt.xlabel("Episode / Map")
    plt.ylabel("Pass (1/0)")
    plt.legend()
    plt.title("Pass Rate")

    plt.tight_layout()
    plt.show()

    # ======= Сохранение модели =======
    agent.save("trained_pathfinder_dqn.pth")
