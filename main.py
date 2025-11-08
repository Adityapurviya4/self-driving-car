import os
import numpy as np
from rl_model.dqn_agent import DQNAgent
from environment.map import RoadMap
import argparse


os.makedirs("data/checkpoints", exist_ok=True)

def train_car(episodes=500, max_steps=1000):

    env = RoadMap()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_steps:

            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            total_reward += reward

            agent.train(state, action, reward, next_state, done)

            state = next_state
            step += 1

        print(f"Episode {episode}/{episodes} | Total Reward: {total_reward}")

        if episode % 50 == 0:
            agent.save_model(f"data/checkpoints/dqn_episode_{episode}.pth")
            print(f"Checkpoint saved at episode {episode}!")

    print("Training complete!")
    env.close()

def run_simulation():

    env = RoadMap()
    done = False
    state = env.reset()

    while not done:
        action = env.random_action()
        state, reward, done = env.step(action)

    env.close()
    print("Simulation finished!")

def main():
    parser = argparse.ArgumentParser(description="Self-Driving Car Simulator with DQN")
    parser.add_argument('--train', action='store_true', help="Train the DQN agent")
    parser.add_argument('--episodes', type=int, default=500, help="Number of training episodes")
    args = parser.parse_args()

    if args.train:
        train_car(episodes=args.episodes)
    else:
        run_simulation()

if __name__ == "__main__":
    main()
