import random
import numpy as np
import pygame
import gym
import tensorflow
from collections import deque
from tensorflow import keras


def feed_replay(nn, target, memory):
    samples = np.zeros(5)
    rewards = np.zeros(5)
    global epsilon
    q_lr = 0.55
    discount = 0.9975
    size = 2000
    batch = random.sample(memory, size)
    output_layer = nn.predict(np.array([i[0].flatten() for i in batch]), batch_size=size//5, verbose=0)
    # print(output_layer)
    next_action_output_layer = target.predict(np.array([i[4].flatten() for i in batch]), batch_size=size//5, verbose=0)
    for index, (first_observation, neuron, reward, terminated, observation) in enumerate(batch):
        rewards[neuron] += reward
        samples[neuron] += 1
        if not terminated:
            q_value = reward + discount * np.max(next_action_output_layer[index])
        else:
            q_value = reward
        output_layer[index][neuron] = (1 - q_lr) * output_layer[index][neuron] + q_lr * q_value
    states = [np.array([i[0].flatten() for i in batch])]
    # print(output_layer)
    # print("max",[np.max(i) for i in next_action_output_layer])
    print(output_layer)
    print(rewards/samples)
    nn.fit(states, output_layer, batch_size=size//5, verbose=0, shuffle=True)
    if epsilon > 0.05:
        epsilon *= 0.925


def make_action(action):
    actions = [0, 0, 0]
    if action < 2 * no_sections + 1:
        actions[0] = -1 + 1 / no_sections * action
    elif action < 3 * no_sections + 1:
        action -= 2 * no_sections
        actions[1] = 1 / no_sections * action
    else:
        action -= 3 * no_sections
        actions[2] = 1 / no_sections * action
    return actions


if __name__ == '__main__':
    no_sections = 1
    print([make_action(i) for i in range(5)])
    global epsilon
    epsilon = 1
    nn_lr = 0.001
    env = gym.make("CarRacing-v2", render_mode="human")
    observation, info = env.reset()
    observation = observation.flatten()
    nn = keras.Sequential()
    nn.add(keras.layers.Dense(200, activation='relu', input_shape=observation.shape, kernel_initializer=keras.initializers.RandomUniform(minval=-0.0005, maxval=0.0005)))
    nn.add(keras.layers.Dense(40, activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.0005, maxval=0.0005)))
    nn.add(keras.layers.Dense(4 * no_sections + 1, activation='linear', kernel_initializer=keras.initializers.RandomUniform(minval=-0.0005, maxval=0.0005)))
    target = keras.Sequential()
    target.add(keras.layers.Dense(200, activation='relu', input_shape=observation.shape, kernel_initializer=keras.initializers.RandomUniform(minval=-0.0005, maxval=0.0005)))
    target.add(keras.layers.Dense(40, activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.0005, maxval=0.0005)))
    target.add(keras.layers.Dense(4 * no_sections + 1, activation='linear', kernel_initializer=keras.initializers.RandomUniform(minval=-0.0005, maxval=0.0005)))
    optimizer = keras.optimizers.Adam(nn_lr)
    nn.compile(optimizer=optimizer, loss='huber', metrics=['accuracy'])
    target.compile(optimizer=optimizer, loss='huber', metrics=['accuracy'])
    target.set_weights(nn.get_weights())
    no_episodes = 500
    memory = deque(maxlen=20_000)
    steps = 0
    remaining_random_actions = 0
    remaining_predicted_actions = 0
    updates = 0
    observation, info = env.reset()
    for i in range(no_episodes):
        print("episode:", i)
        steps_wo_reward = 0
        while True:
            steps += 1
            print(epsilon, steps_wo_reward, len(memory), steps)
            if remaining_random_actions > 0:
                remaining_random_actions -= 1
                neuron = chosen_random_action
            elif remaining_predicted_actions > 0:
                remaining_predicted_actions -= 1
                neuron = predicted_action
            else:
                if random.random() < epsilon:
                    remaining_random_actions = 0
                    neuron = random.randint(0, 4 * no_sections)
                    chosen_random_action = neuron
                else:
                    output_layer = nn.predict(np.array([observation.flatten()]))
                    remaining_predicted_actions = 0
                    neuron = np.argmax(output_layer)
                    predicted_action = neuron
            first_observation = observation
            observation, reward, terminated, truncated, info = env.step(make_action(neuron))
            if reward > 0:
                steps_wo_reward = 0
            else:
                steps_wo_reward += 1
                if steps_wo_reward > 200:
                    reward -= 100
                    memory.append((first_observation, neuron, reward, terminated, observation))
                    break
            memory.append((first_observation, neuron, reward, terminated, observation))
            if terminated:
                break
            if updates == 15:
                target.set_weights(nn.get_weights())
                updates = 0
            if steps % 250 == 0 and len(memory) > 2000:
                feed_replay(nn, target, memory)
                updates += 1
        observation, info = env.reset()
    env = gym.make("CarRacing-v2", render_mode="human")
    observation, info = env.reset()
    while True:
        output_layer = nn.predict(np.array([observation.flatten()]))
        neuron = np.argmax(output_layer)
        q_value = output_layer[0][neuron]
        # print(output_layer)
        observation, reward, terminated, truncated, info = env.step(make_action(neuron))
        if terminated:
            observation, info = env.reset()
