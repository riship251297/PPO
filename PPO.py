import gym
from torch.distributions import MultivariateNormal

gym.logger.set_level(40)

import pybullet as p
import pybullet_envs

# Models and computation
import torch  # will use pyTorch to handle NN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from random import sample

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from numpngw import write_apng

env = gym.make("HopperBulletEnv-v0")
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]


class ACTOR_NETWORK(nn.Module):
    def __init__(self, NUM_STATES, NUM_ACTIONS, FULLY_CONNECTED_LAYER_1=32, FULLY_CONNECTED_LAYER_2=32):
        super(ACTOR_NETWORK, self).__init__()

        self.FULLY_CONNECTED_LAYER_1 = nn.Linear(NUM_STATES, FULLY_CONNECTED_LAYER_1)
        self.FULLY_CONNECTED_LAYER_2 = nn.Linear(FULLY_CONNECTED_LAYER_1, FULLY_CONNECTED_LAYER_2)
        self.FULLY_CONNECTED_LAYER_3 = nn.Linear(FULLY_CONNECTED_LAYER_2, NUM_ACTIONS)

        self.ACTION_SELECTION_VARIANCE = torch.full((NUM_ACTIONS,), 0.05)

    def forward(self, STATE_INPUT):
        ACTIVATION_1 = F.relu(self.FULLY_CONNECTED_LAYER_1(STATE_INPUT))
        ACTIVATION_2 = F.relu(self.FULLY_CONNECTED_LAYER_2(ACTIVATION_1))
        OUTPUT = torch.tan(self.FULLY_CONNECTED_LAYER_3(ACTIVATION_2))
        return OUTPUT


class CRITIC_NETWORK(nn.Module):
    def __init__(self, NUM_STATES, FULLY_CONNECTED_LAYER_1=32, FULLY_CONNECTED_LAYER_2=32):
        super(CRITIC_NETWORK, self).__init__()
        self.FULLY_CONNECTED_LAYER_1 = nn.Linear(NUM_STATES, FULLY_CONNECTED_LAYER_1)
        self.FULLY_CONNECTED_LAYER_2 = nn.Linear(FULLY_CONNECTED_LAYER_1, FULLY_CONNECTED_LAYER_2)
        self.FULLY_CONNECTED_LAYER_3 = nn.Linear(FULLY_CONNECTED_LAYER_2, 1)

    def forward(self, STATES_INPUT):
        states_y = torch.unsqueeze(torch.FloatTensor(STATES_INPUT), 0)
        ACTIVATION_1 = F.relu(self.FULLY_CONNECTED_LAYER_1(states_y))
        ACTIVATION_2 = F.relu(self.FULLY_CONNECTED_LAYER_2(ACTIVATION_1))
        OUTPUT = self.FULLY_CONNECTED_LAYER_3(ACTIVATION_2)
        return OUTPUT[0]


class PROXIMAL_POLICY_OPTIMIZATION:
    def __init__(self, policy_learning_rate=5e-4, value_function_learning_rate=1e-2):

        self.log_prob_actions = []
        self.ACTIONS = []
        self.STATES = []
        self.total_steps = 0
        self.episode_reward = []
        self.evaluation_scores = []
        self.gamma = 0.99
        self.log_probs = []

        self.POLICY_NETWORK = ACTOR_NETWORK(NUM_STATES, NUM_ACTIONS)

        self.VALUE_NETWORK = CRITIC_NETWORK(NUM_STATES)

        self.policy_optimizer = optim.Adam(self.POLICY_NETWORK.parameters(), lr=policy_learning_rate)
        self.value_optimizer = optim.Adam(self.VALUE_NETWORK.parameters(), lr=value_function_learning_rate)

    # <------------------------------- ACTION-SELECTION USING STOCHASTIC POLICY ------------------------------------>

    def action_selection(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        self.STATES.append(state)

        ACTION_MEAN = self.POLICY_NETWORK.forward(state)
        COVARIANCE_MATRIX = torch.diag(self.POLICY_NETWORK.ACTION_SELECTION_VARIANCE)
        dist = MultivariateNormal(ACTION_MEAN, COVARIANCE_MATRIX)
        ACTION = dist.sample()

        LOG_PROB_ACTION = dist.log_prob(ACTION)
        ACTION = ACTION.detach().numpy()
        ACTION = np.squeeze(np.clip(ACTION, -1, 1))
        ACTION_TO_BE_APPENDED = torch.Tensor(ACTION)
        self.ACTIONS.append(ACTION_TO_BE_APPENDED)

        return ACTION, LOG_PROB_ACTION, self.STATES, self.ACTIONS

    # <------------------------------- GREEDY-ACTION-SELECTION USING DETERMINISTIC POLICY ------------------------------------>

    def select_greedy_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        GREEDY_ACTION = self.POLICY_NETWORK.forward(state)
        GREEDY_ACTION = GREEDY_ACTION.detach().numpy()
        GREEDY_ACTION = np.squeeze(np.clip(GREEDY_ACTION, -1, 1))

        return GREEDY_ACTION

    # <------------------------------- POLICY AND VALUE LOSS UPDATE MODULE USING CLIPPED OBJECTIVE  ------------------------------------>

    def update(self, STATES, ACTIONS, OLD_LOG_PROB_ACTIONS, ADVANTAGES, TARGETS):
        STATES = STATES.detach()
        ACTIONS = ACTIONS.detach()
        OLD_LOG_PROB_ACTIONS = OLD_LOG_PROB_ACTIONS.detach()
        ADVANTAGES = ADVANTAGES.unsqueeze(1).detach()
        TARGETS = TARGETS.detach()
        EPOCHS = 50

        for i in range(EPOCHS):
            ACTION_MEAN = self.POLICY_NETWORK.forward(STATES)
            VALUE_STATE = self.VALUE_NETWORK.forward(STATES)

            ACTION_VARIANCE = self.POLICY_NETWORK.ACTION_SELECTION_VARIANCE.expand_as(ACTION_MEAN)
            cov_mat = torch.diag_embed(ACTION_VARIANCE)
            dist = MultivariateNormal(ACTION_MEAN, cov_mat)
            NEW_LOG_PROB_ACTIONS = dist.log_prob(ACTIONS)
            ENTROPY = dist.entropy()

            CLIP_VALUE = 0.2

            # <------------------------------- CALCULATION OF POLICY LOSS USING CLIPPED OBJECTIVE  ------------------------------------>

            RATIO = torch.exp(NEW_LOG_PROB_ACTIONS - OLD_LOG_PROB_ACTIONS)

            SURROGATE_1 = RATIO * ADVANTAGES

            SURROGATE_2 = torch.clamp(RATIO, min=1.0 - CLIP_VALUE, max=1.0 + CLIP_VALUE) * ADVANTAGES

            POLICY_LOSS = - torch.min(SURROGATE_1, SURROGATE_2).mean() - 0.001 * ENTROPY.mean()

            self.policy_optimizer.zero_grad()
            POLICY_LOSS.backward()
            self.policy_optimizer.step()
            # <------------------------------- CALCULATION OF VALUE LOSS  ------------------------------------>

            VALUE_LOSS = (VALUE_STATE - TARGETS).pow(2).mean()

            self.value_optimizer.zero_grad()
            VALUE_LOSS.backward()
            self.value_optimizer.step()

            # <------------------------------- TRAINING MODULE ------------------------------------>

    def TRAINING_MODULE(self):
        result = np.empty((1000, 2))

        # <------------------------------- EPISODE STARTS ---------------------------------------->

        for episode in range(0, 1000):
            self.VALUES_LIST = []
            self.REWARDS_LIST = []
            self.entropies = []

            episode_steps = 0
            self.episode_reward.append(0.0)
            state, is_terminal = env.reset(), False

            while True:
                ACTION, LOG_PROB_ACTION, STATES, ACTIONS = self.action_selection(state)
                new_state, reward, is_done, _ = env.step(ACTION)

                self.log_prob_actions.append(LOG_PROB_ACTION)
                self.REWARDS_LIST.append(reward)

                self.VALUES_LIST.append(self.VALUE_NETWORK(state))
                self.episode_reward[-1] += reward
                self.total_steps += 1
                episode_steps += 1

                state = new_state

                timeout = episode_steps == env.spec.max_episode_steps
                if is_done or timeout:
                    if timeout:
                        next_value = self.VALUE_NETWORK(state).detach().item()
                    else:
                        next_value = 0

                    self.VALUES_LIST.append(torch.FloatTensor([next_value, ], ))
                    self.REWARDS_LIST.append(next_value)
                    self.VALUES_LIST = torch.cat(self.VALUES_LIST)

                    self.np_values = self.VALUES_LIST.view(-1).data.numpy()

                    self.ADVANTAGE = 0
                    ADVANTAGES = []
                    TARGETS = []
                    NEXT_VALUE = 0

                    # <------------------------------- CALCULATION OF ADVANTAGE USING GENERALIZED ADVANTAGE ESTIMATE ------------------------------------>

                    for REWARDS, OLD_VALUE in zip(reversed(self.REWARDS_LIST), reversed(self.np_values)):
                        TEMPORAL_DIFFERENCE_ERROR = REWARDS + self.gamma * NEXT_VALUE - OLD_VALUE
                        self.ADVANTAGE = TEMPORAL_DIFFERENCE_ERROR + self.ADVANTAGE * self.gamma * 0.99

                        # <------------------------------- CALCULATION OF TARGET ---------------------------------------->

                        target = self.ADVANTAGE + OLD_VALUE
                        NEXT_VALUE = OLD_VALUE

                        ADVANTAGES.append(self.ADVANTAGE)
                        TARGETS.append(target)
                    break

                    # <------------------------------- BUFFER ---------------------------------------->

            STATES_BUFFER = torch.squeeze(torch.stack(STATES))
            ACTIONS_BUFFER = torch.squeeze(torch.stack(ACTIONS))
            OLD_LOG_PROB_BUFFER = torch.squeeze(torch.stack(self.log_prob_actions))

            TARGETS = torch.tensor(TARGETS)

            ADVANTAGES = torch.Tensor(ADVANTAGES)

            self.update(STATES_BUFFER, ACTIONS_BUFFER, OLD_LOG_PROB_BUFFER, ADVANTAGES, TARGETS)

            evaluation_score = self.evaluate()
            self.evaluation_scores.append(evaluation_score)

            result[episode - 1] = evaluation_score, self.total_steps

            print('Iteration {}\tEvaluation Score: {:.2f}\tAvg. Evaluation Scores: {:.2f}\tSteps: {}'.format(episode,
                                                                                                            evaluation_score,
                                                                                                            np.mean(
                                                                                                                self.evaluation_scores),
                                                                                                            self.total_steps))

            if episode > 1000:
                break

        save_checkpoint(self.POLICY_NETWORK, 'POLICY.pt', 1)
        save_checkpoint(self.VALUE_NETWORK, 'VALUE.pt', 1)

        return np.array(result)

    # <------------------------------- ROLLOUT  ---------------------------------------->

    def evaluate(self):
        rewards = []
        env = gym.make("HopperBulletEnv-v0")
        for i in range(1):
            state, done = env.reset(), False
            rewards.append(0)
            while not done:
                GREEDY_ACTION = self.select_greedy_action(state)
                state, reward, done, _ = env.step(GREEDY_ACTION)
                rewards[-1] += reward

        return np.mean(rewards)


if __name__ == '__main__':
    AGENT = PROXIMAL_POLICY_OPTIMIZATION()
    result = AGENT.TRAINING_MODULE()
    eval_rewards, steps = result.T

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    plt.plot(steps, eval_rewards)
    plt.ylabel('REWARDS')
    plt.xlabel('NUMBER_OF_STEPS')
    leg = plt.legend()
    ax2.set_title('Evaluation Reward ')
    plt.show()
    env.close()
