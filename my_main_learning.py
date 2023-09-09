""" Upon this rock, I will build my church """
from environment import Env
import numpy as np
from ppo_learning import Agent
import matplotlib.pyplot as plt
import serial
from torch.utils.tensorboard import SummaryWriter
from playsound import playsound
import os

arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)


def write(x):
    arduino.write(bytes(x, 'utf-8'))
    # time.sleep(0.001)  # remove if ppo is established


def read():
    return arduino.readline()


def interpolate(x, x1, x2, y1, y2):
    return y1 + (x - x1) * ((y2 - y1)/(x2 - x1))


def mod_obs(obs):
    """
    angle:
    (-0.4..., -0.011], (-0.011, -0.004], (-0.004, 0.002), [0, 0.007), [0.007, 0.4...)
    angular velocity:
    [-10..., -0.5], (-0.5, -0.16], (-0.16, 0.16), [0.16, 0.5), and [0.5, 10...]
    """
    new_observation = [0. for _ in range(10)]
    angle = obs[1]
    if angle <= -.011:
        new_observation[0] = interpolate(angle, -0.011, -0.4, 0, 1)
    elif -0.011 < angle <= -.004:
        new_observation[1] = interpolate(angle, -.004, -.011, 0, 1)
    elif -.002 < angle < 0.002:
        new_observation[2] = interpolate(angle, -.002, 0.002, -1, 1)
    elif 0. <= angle < 0.007:
        new_observation[3] = interpolate(angle, 0., 0.007, 0, 1)
    else:
        new_observation[4] = interpolate(angle, 0.007, 0.4, 0, 1)

    velo = obs[0]
    if velo <= -0.5:
        new_observation[5] = interpolate(velo, -0.5, -10, 0, 1)
    elif -0.5 < velo <= -.16:
        new_observation[6] = interpolate(velo, -.16, -0.5, 0, 1)
    elif -.16 < velo < .16:
        new_observation[7] = interpolate(velo, -.16, .16, -1, 1)
    elif .16 <= velo < 0.5:
        new_observation[8] = interpolate(velo, .16, 0.5, 0, 1)
    else:
        new_observation[9] = interpolate(velo, 0.5, 10, 0, 1)
    return new_observation


def decode(line):
    elements = line.decode('utf-8')[:-2].split(',')
    obs = [float(element) for element in elements[:2]]
    # obs = mod_obs(obs)
    return obs, 1, bool(int(elements[2]))


def init_env():
    print("Establishing Connection....")
    while True:
        text = read()
        if text == b'ready!\r\n':
            print(text.decode('utf-8'))
            break
        elif text != b'':
            print(text)


def mod_action(env, action):
    """
    0: -20,  1: -10,  2: -6,  3: -3,  4: 0,  5: 3,  6: 6,  7: 10,  8: 20
    """
    dir = 1 if action > 4 else 0
    if action == 4:
        env.force_mag = 0
    elif action in [3, 5]:
        env.force_mag = 3
    elif action in [2, 6]:
        env.force_mag = 6
    elif action in [1, 7]:
        env.force_mag = 12
    elif action in [0, 8]:
        env.force_mag = 20
    return env.step(dir)


def agent_play(env, agent, eps=10):
    for _ in range(eps):
        observation = mod_obs(env.reset())
        done = False
        while not done:
            env.render()
            action, prob, val = agent.choose_action(observation)
            observation_, _, done, _ = mod_action(env, action)
            observation_ = mod_obs(observation_)
            observation = observation_
    env.close()


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.clf()


if __name__ == "__main__":
    # make sure to run in ipython console
    init_env()
    horizon = 64
    batch_size = 16
    n_epochs = 30
    alpha = 0.0003
    n_games = 300
    gamma = 0.99
    policy_clip = 0.2
    agent = Agent(n_actions=9, batch_size=batch_size, n_games=n_games, gamma=gamma, policy_clip=policy_clip,
                  alpha=alpha, n_epochs=n_epochs, input_dims=(2,))
    # agent.load_models(lr=0.00024119999999999868, clip=0.17386666666666684)
    agent.actor.checkpoint_file = "./models/actor_torch_ppo_relearn_3"
    agent.critic.checkpoint_file = "./models/critic_torch_ppo_relearn_3"
    agent.load_models()
    agent.actor.checkpoint_file = "./tmp/ppo_learn/actor_torch_ppo_learn"
    agent.critic.checkpoint_file = "./tmp/ppo_learn/critic_torch_ppo_learn"
    writer = SummaryWriter(log_dir=f"runs/robot/train 21,new_params,256 neurons")
    best_score = 0
    score_history = []
    losses = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    loss_ = 0
    max_steps = 1000

    i = 0
    print('Start Episode Now')
    while True:
        text = read()
        if text == b'start\r\n':
            steps = 0
            episode_data = []
            text = read()
            observation, _, done = decode(text)
            score = 0
            loss = 0
            while True:
                text = read()
                if text == b'end\r\n':
                    break
                elif text == b'':
                    pass
                else:
                    action, prob, val = agent.choose_action(observation)
                    # print(action)
                    write(str(action))
                    observation_, reward, done = decode(text)
                    score += reward
                    episode_data.append((observation, action, prob, val, reward, done))
                    steps += 1
                    if steps > max_steps:
                        break
                    observation = observation_

            if len(episode_data) > 5:
                loss_count = 0
                loss = 0
                for data in episode_data:
                    agent.remember(*data)
                    n_steps += 1
                    if n_steps % horizon == 0:
                        loss += agent.learn()
                        n_steps = 0
                        loss_count += 1
                        learn_iters += 1
                if loss_count > 0:
                    loss /= loss_count
                else:
                    loss = loss_
                agent.decay_lr()
                agent.decay_clip()
                # agent.decay_epoch()
                writer.add_scalar("train/reward", score, i)
                writer.add_scalar("train/loss", loss, i)
                writer.add_scalar("train/learning rate", agent.alpha, i)
                writer.add_scalar("train/policy clip", agent.policy_clip, i)
                # writer.add_scalar("train/epochs", agent.n_epochs, i)
                writer.flush()
                score_history.append(score)
                losses.append(loss)
                loss_ = loss
                avg_score = np.mean(score_history[-100:])

                if avg_score > best_score:
                    best_score = avg_score
                    agent.save_models()

                print("episode", i, "learning_steps", learn_iters, "score %.1f" % score, "avg score %.1f" % avg_score,
                      "learning_rate %.6f" % agent.alpha, "loss %.6f" % loss)
                with open(f"./data/episode_{i}.txt", 'w') as file:
                    file.write('\n'.join([str(ep) for ep in episode_data]))
                playsound(os.path.join(os.path.dirname(__file__), 'ding.mp3'))
                i += 1
                if i > n_games:
                    break
            else:
                print("Episode Skipped")

    agent.save_models()
    x1 = [i + 1 for i in range(len(losses))]
    x2 = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x1, losses, 'plots/learn_reward_losses.png')
    plot_learning_curve(x2, score_history, 'plots/learn_reward_scores.png')
    writer.close()
