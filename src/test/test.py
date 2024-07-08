import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate sample data
num_points_time = 120  # Two minutes of data
num_points_frequency = 50
# Random magnitude data in dB
data = np.random.rand(num_points_time * 10, num_points_frequency) * 100

# Create a figure and axis
fig, ax = plt.subplots()

# Create initial colorbar
cbar = plt.colorbar(ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest', origin='lower',
                              extent=[0, num_points_frequency, 0, num_points_time]), ax=ax)
cbar.set_label('Magnitude (dB)')

# Function to update the plot


def update(frame):
    ax.clear()

    print(data.shape)
    # Calculate the starting index for the last two minutes of data
    start_index = max(0, frame - num_points_time)
    im = ax.imshow(data[start_index:frame], cmap='viridis', aspect='auto', interpolation='nearest', origin='lower',
                   extent=[0, num_points_frequency, start_index, frame])  # Displaying the waterfall plot
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Time')
    ax.set_title('Waterfall Plot')
    ax.set_xlim(0, num_points_frequency)
    ax.set_ylim(max(0, frame - num_points_time), frame)
    print(start_index, frame)


# Create animation
ani = FuncAnimation(fig, update, repeat=False)

plt.show()
