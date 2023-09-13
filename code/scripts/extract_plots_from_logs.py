import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse

def main(*args):
    log_dir = args[0].log_dir
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extract scalar data
    scalar_tags = event_acc.Tags()['scalars']
    scalar_data = {}
    for tag in scalar_tags:
        scalar_data[tag] = event_acc.Scalars(tag)

    # Define the monitored scalar tags in the desired order
    monitored_scalar_tags = [
        'Loss/color_loss',
        'Loss/eikonal_loss',
        'Loss/Loss',
        'Loss/mask_loss'
    ]
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Monitored Scalars Visualization', fontsize=16)

    # Plot the monitored scalar values in the subplots
    for i, tag in enumerate(monitored_scalar_tags):
        data = scalar_data.get(tag)
        if data:
            steps = [step.step for step in data]
            values = [step.value for step in data]
            axs[i // 2, i % 2].plot(steps, values)
            axs[i // 2, i % 2].set_title(tag)
            axs[i // 2, i % 2].set_xlabel('EpochStep')
            axs[i // 2, i % 2].set_ylabel('LossVal')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    Argaparser = argparse.ArgumentParser()
    Argaparser.add_argument("--log_dir", type=str, default="logs", help="log directory")
    *args, = Argaparser.parse_args()
    main(*args)
     