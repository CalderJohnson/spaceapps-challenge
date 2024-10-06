"""Create a plot of seismic data indicating predicted quake."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_plot(file, prediction, truth=None):
    """Generate plot."""
    df = pd.read_csv(file)
    
    # Read in time steps and velocities
    csv_times = np.array(df['time_rel(sec)'].tolist())
    csv_data = np.array(df['velocity(m/s)'].tolist())

    # Plot the trace
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(csv_times, csv_data)

    # Make the plot pretty
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{file}', fontweight='bold')

    # Plot where the arrival time is
    arrival_line = ax.axvline(x=prediction, c='red', label='Rel. Arrival')
    ax.legend(handles=[arrival_line])

    # Plot the true arrival time if available
    if truth:
        truth_line = ax.axvline(x=truth, c='green', label='True Arrival')
        ax.legend(handles=[arrival_line, truth_line])

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Get the current working directory (assumed to be the project root)
    project_root = os.getcwd()

    # Create 'plots' directory in the project root if it doesn't exist
    stuff_dir = os.path.join(project_root, 'plots')
    if not os.path.exists(stuff_dir):
        os.makedirs(stuff_dir)

    # Save the plot as PNG in the 'plots' folder
    plot_path = os.path.join(stuff_dir, 'lunar_seismic_event_plot.png')
    plt.savefig(plot_path)

    print(f"Plot saved as '{plot_path}'")
