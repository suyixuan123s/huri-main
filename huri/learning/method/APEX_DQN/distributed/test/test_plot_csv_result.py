""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231102osaka

"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation_metrics(csv_paths: str or list):
    # If a single string is given, convert it to a list
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    # Get the number of files
    num_files = len(csv_paths)

    # Setting up the figure and subplots
    fig, axes = plt.subplots(nrows=3, ncols=num_files, figsize=(12 * num_files, 15))

    # If there's only one file, ensure axes is a 2D array for consistency
    if num_files == 1:
        axes = axes.reshape(3, 1)

    # Set the global font
    plt.rcParams["font.family"] = "Times New Roman"

    for idx, csv_path in enumerate(csv_paths):
        # Load the CSV file into a DataFrame
        eval_df = pd.read_csv(csv_path)

        # 1. Relationship between train_steps and eval_average_len/eval_average_score
        axes[0, idx].plot(eval_df['train_steps'], eval_df['eval_average_len'], label='Average Length', color='blue')
        axes[0, idx].plot(eval_df['train_steps'], eval_df['eval_average_score'], label='Average Score', color='red')
        axes[0, idx].set_title(f'Train Steps vs. Evaluation Average Length & Score ({csv_path.split("/")[-1]})')
        axes[0, idx].set_xlabel('Train Steps')
        axes[0, idx].set_ylabel('Value')
        axes[0, idx].legend()

        # 2. Relationship between train_steps and eval_num_dones
        axes[1, idx].plot(eval_df['train_steps'], eval_df['eval_num_dones'], color='green')
        axes[1, idx].set_title(f'Train Steps vs. Evaluation Number of Dones ({csv_path.split("/")[-1]})')
        axes[1, idx].set_xlabel('Train Steps')
        axes[1, idx].set_ylabel('Number of Dones')

        # 3. Relationship between train_steps and training_level
        axes[2, idx].plot(eval_df['train_steps'], eval_df['training_level'], color='orange')
        axes[2, idx].set_title(f'Train Steps vs. Training Level ({csv_path.split("/")[-1]})')
        axes[2, idx].set_xlabel('Train Steps')
        axes[2, idx].set_ylabel('Training Level')

    # Adjusting the layout
    plt.tight_layout()
    plt.show()


def plot_evaluation_metrics2(*csv_paths):
    # Setting up the figure and subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))

    # Set the global font
    plt.rcParams["font.family"] = "Times New Roman"

    for csv_path in csv_paths:
        # Load the CSV file into a DataFrame
        eval_df = pd.read_csv(csv_path)

        # 1. Relationship between train_steps and eval_average_len/eval_average_score
        # axes[0].plot(eval_df['train_steps'], eval_df['eval_average_len'], label=f'Average Length ({csv_path})')
        # axes[0].plot(eval_df['train_steps'], eval_df['eval_average_score'], label=f'Average Score ({csv_path})')
        axes[0].plot(eval_df['train_steps'], eval_df['eval_average_len'], label=f'Average Length')
        axes[0].plot(eval_df['train_steps'], eval_df['eval_average_score'], label=f'Average Score')

        # 2. Relationship between train_steps and eval_num_dones
        axes[1].plot(eval_df['train_steps'], eval_df['eval_num_dones'], label=f'{csv_path}')

        # 3. Relationship between train_steps and training_level
        axes[2].plot(eval_df['train_steps'], eval_df['training_level'], label=f'{csv_path}')

    # Setting titles, labels, legends for all subplots
    axes[0].set_title('Train Steps vs. Evaluation Average Length & Score')
    axes[0].set_xlabel('Train Steps')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    axes[1].set_title('Train Steps vs. Evaluation Number of Dones')
    axes[1].set_xlabel('Train Steps')
    axes[1].set_ylabel('Number of Dones')
    axes[1].legend()

    axes[2].set_title('Train Steps vs. Training Level')
    axes[2].set_xlabel('Train Steps')
    axes[2].set_ylabel('Training Level')
    axes[2].legend()

    # Adjusting the layout
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_evaluation_metrics([r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp4\run\log\eval_log.csv',
                             r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp4\run2\log\eval_log.csv',
                             r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\log\eval_log.csv'])
