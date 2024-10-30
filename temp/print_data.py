import matplotlib.pyplot as plt
import re
import numpy as np


# Function to extract data from a file
def extract_data(filename):
    pattern = r"episode (\d+) score (-?\d+\.\d+) avg score (-?\d+\.\d+) critic loss (-?\d+\.\d+) actor loss (-?\d+\.\d+) system loss (-?\d+\.\d+) reward loss (-?\d+\.\d+)"
    data = []
    with open(filename, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append([float(x) for x in match.groups()[1:]])
    return data


# Function to compute average scores
def compute_avg_scores(scores, window_size=100):
    return [
        np.mean(scores[max(0, i - window_size + 1) : i + 1]) for i in range(len(scores))
    ]


# Function to plot data
def plot_data(data_a, data_b, show_single=False, single_dataset=None):
    (
        scores_a,
        _,
        critic_losses_a,
        actor_losses_a,
        system_losses_a,
        reward_losses_a,
    ) = zip(*data_a)
    (
        scores_b,
        _,
        critic_losses_b,
        actor_losses_b,
        system_losses_b,
        reward_losses_b,
    ) = zip(*data_b)

    avg_scores_a = compute_avg_scores(scores_a)
    avg_scores_b = compute_avg_scores(scores_b)

    if show_single and single_dataset is not None:
        (
            single_scores,
            single_avg_scores,
            single_critic_losses,
            single_actor_losses,
            single_system_losses,
            single_reward_losses,
        ) = (None, None, None, None, None, None)
        if single_dataset == "A":
            (
                single_scores,
                single_avg_scores,
                single_critic_losses,
                single_actor_losses,
                single_system_losses,
                single_reward_losses,
            ) = (
                scores_a,
                avg_scores_a,
                critic_losses_a,
                actor_losses_a,
                system_losses_a,
                reward_losses_a,
            )
        elif single_dataset == "B":
            (
                single_scores,
                single_avg_scores,
                single_critic_losses,
                single_actor_losses,
                single_system_losses,
                single_reward_losses,
            ) = (
                scores_b,
                avg_scores_b,
                critic_losses_b,
                actor_losses_b,
                system_losses_b,
                reward_losses_b,
            )

        print(f"Scores: {single_scores}")
        print(f"Avg Scores: {single_avg_scores}")
        print(f"Critic Losses: {single_critic_losses}")
        print(f"Actor Losses: {single_actor_losses}")
        print(f"System Losses: {single_system_losses}")
        print(f"Reward Losses: {single_reward_losses}")

    fig, axs = plt.subplots(6, figsize=(10, 20))

    axs[0].plot(scores_a, label="Scores A")
    axs[0].plot(scores_b, label="Scores B", linestyle="--")
    axs[0].set_title("Scores")
    axs[0].legend()

    axs[1].plot(avg_scores_a, label="Avg Scores A")
    axs[1].plot(avg_scores_b, label="Avg Scores B", linestyle="--")
    axs[1].set_title("Average Scores")
    axs[1].legend()

    axs[2].plot(critic_losses_a, label="Critic Losses A")
    axs[2].plot(critic_losses_b, label="Critic Losses B", linestyle="--")
    axs[2].set_title("Critic Losses")
    axs[2].legend()

    axs[3].plot(actor_losses_a, label="Actor Losses A")
    axs[3].plot(actor_losses_b, label="Actor Losses B", linestyle="--")
    axs[3].set_title("Actor Losses")
    axs[3].legend()

    axs[4].plot(system_losses_a, label="System Losses A")
    axs[4].plot(system_losses_b, label="System Losses B", linestyle="--")
    axs[4].set_title("System Losses")
    axs[4].legend()

    axs[5].plot(reward_losses_a, label="Reward Losses A")
    axs[5].plot(reward_losses_b, label="Reward Losses B", linestyle="--")
    axs[5].set_title("Reward Losses")
    axs[5].legend()

    fig.tight_layout()
    plt.show()


# Extract data from two different files
data_a = extract_data("data_c_0.4beta_0.1polnoise.txt")
data_b = extract_data("data_a_0.4beta.txt")

# Call plot_data with show_single as True and single_dataset as 'A' or 'B'
plot_data(data_a, data_b, show_single=False, single_dataset="A")
