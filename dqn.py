import argparse
import os
import random
import time
from typing import NamedTuple

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vessl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from torch.utils.tensorboard.writer import SummaryWriter


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBufferSample(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    terminateds: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        np_rng: np.random.Generator,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.np_rng = np_rng

        self.pos = 0
        self.full = False

        # Initialize buffers
        self.obs_shape = self._get_obs_shape(self.observation_space)
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=self.observation_space.dtype
        )

        self.action_dim = self._get_action_dim(self.action_space)
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=self.action_space.dtype
        )

        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.terminateds = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
    ):
        # Store a data
        self.observations[self.pos] = np.array(observation).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = reward
        self.terminateds[self.pos] = terminated

        # Update the index
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        if self.full:
            batch_inds = (
                self.np_rng.integers(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = self.np_rng.integers(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_obs_shape(self, observation_space: gym.Space) -> tuple[int, ...]:
        if isinstance(observation_space, spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, spaces.Discrete):
            return (1,)
        else:
            raise NotImplementedError(
                f"{observation_space} observation space is not supported."
            )

    def _get_action_dim(self, action_space: gym.Space) -> int:
        if isinstance(action_space, spaces.Box):
            return int(np.prod(action_space.shape))
        elif isinstance(action_space, spaces.Discrete):
            return 1
        else:
            raise NotImplementedError(f"{action_space} action space is not supported.")

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSample:
        next_obs = self.observations[(batch_inds + 1) % self.buffer_size, :]
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            next_obs,
            self.terminateds[batch_inds],
            self.rewards[batch_inds],
        )
        return ReplayBufferSample(*tuple(map(self._to_torch, data)))

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-id",  type=str, default="CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-network-frequency", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-eps", type=float, default=1.0)
    parser.add_argument("--end-eps", type=float, default=0.05)
    parser.add_argument("--eps-fraction", type=float, default=0.5)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--train-frequency", type=int, default=10)
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(args, run_name):
    env = gym.make("CartPole-v1")
    env = RecordEpisodeStatistics(env)
    if args.record_video:
        env = RecordVideo(env, f"videos/{run_name}")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


def linear_schedule(start_eps, end_eps, duration, t):
    slope = (end_eps - start_eps) / duration
    return max(slope * t + start_eps, end_eps)


def main():
    args = parse_args()

    # Setup VESSL
    vessl.init(tensorboard=True)

    # Setup Tensorboard
    file_name = os.path.basename(__file__).rstrip(".py")
    run_name = f"{args.env_id}__{file_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Log arguments (hyperparameters)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an environment
    env = make_env(args, run_name)

    # Create a DQN model
    q_network = QNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Create a replay buffer (RB)
    rb = ReplayBuffer(
        args.buffer_size, env.observation_space, env.action_space, device, np_rng
    )

    # Environment Loop
    st = time.time()
    obs, _ = env.reset(seed=args.seed)
    for global_t in range(args.total_timesteps):
        # Eps-greedy action selection
        eps = linear_schedule(
            args.start_eps,
            args.end_eps,
            args.eps_fraction * args.total_timesteps,
            global_t,
        )
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.tensor(obs, device=device))
            action = torch.argmax(q_values).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Log episode statistics if available
        if "episode" in info.keys():
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_t)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_t)
            writer.add_scalar("charts/eps", eps, global_t)

        # Update RB and obs
        rb.add(obs, action, reward, terminated)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Training Loop
        if global_t > args.learning_starts and global_t % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards + args.gamma * target_max * (
                    1 - data.terminateds
                )
            prediction = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, prediction)

            # Log train statistics
            if global_t % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_t)
                writer.add_scalar("losses/q_values", prediction.mean().item(), global_t)
                writer.add_scalar(
                    "charts/SPS", int(global_t / (time.time() - st)), global_t
                )

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if global_t % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # Cleanup
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
