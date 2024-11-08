import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def euclidean_distance(lat1, lon1, lat2, lon2):
    """Calculate Euclidean distance between two points (latitude, longitude)."""
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

def animate_user_movement(user_input_file, user_id):
    # Load the dataset
    data = pd.read_csv(user_input_file)

    # Filter data for the specified user
    user_data = data[data['USERID'] == user_id]

    if user_data.empty:
        print(f"No data found for user ID {user_id}.")
        return

    # Sort by timestamp to ensure proper order of movement
    user_data = user_data.sort_values(by='TIMESTAMP')

    # Extract relevant data
    longitudes = user_data['LONGITUDE'].values
    latitudes = user_data['LATITUDE'].values
    space_ids = user_data['SPACEID'].values
    timestamps = user_data['TIMESTAMP'].values

    # Calculate distances and speeds
    distances = np.zeros(len(latitudes) - 1)
    for i in range(len(latitudes) - 1):
        distances[i] = euclidean_distance(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1]) / 1000

    # Calculate time differences in seconds
    time_differences = np.diff(timestamps) / 1000  # Convert to seconds

    # Calculate speeds in km/hr
    speeds = np.zeros_like(distances)
    for i in range(len(time_differences)):
        if time_differences[i] > 0:
            speeds[i] = (distances[i] / time_differences[i]) * 3600  # Speed in km/hr
        else:
            speeds[i] = 0  # No movement

    # Append zeros for the last position (no movement)
    speeds = np.append(speeds, 0)
    distances = np.append(distances, 0)  # Append 0 for the last distance (no movement)

    # Create figure and axis
    fig, ax = plt.subplots()
    scatter = ax.scatter(longitudes[0], latitudes[0], c='blue', s=100)
    text_annotation = ax.text(0, 0, '', fontsize=12, ha='center', va='bottom')

    ax.set_xlim(longitudes.min() - 0.01, longitudes.max() + 0.01)
    ax.set_ylim(latitudes.min() - 0.01, latitudes.max() + 0.01)
    ax.set_title(f"User Movement for User ID {user_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    step_count = 0  # Initialize step count
    step_size = 0.000762  # Example step size in kilometers (average step size ~0.762 m)

    def update(frame):
        nonlocal step_count
        scatter.set_offsets([longitudes[frame], latitudes[frame]])
        text_annotation.set_position((longitudes[frame], latitudes[frame]))

        # Update the step count using the formula
        if frame > 0 and time_differences[frame - 1] > 0:
            distance_travelled = distances[frame - 1]
            time_interval = time_differences[frame - 1]
            steps = distance_travelled / (step_size * time_interval)
            step_count += int(steps)

        text_annotation.set_text(
            f"Space ID: {space_ids[frame]}\nSpeed: {speeds[frame]:.2f} km/hr\nSteps: {step_count}"
        )
        return scatter, text_annotation

    ani = animation.FuncAnimation(fig, update, frames=len(longitudes), interval=10, blit=True, repeat=False)
    plt.show()

# Example usage
user_input_file = r'C:\Users\Hp\OneDrive\Minor Project\archive (1)\TrainingData.csv'  
user_id = int(input("Enter the User ID: "))
animate_user_movement(user_input_file, user_id)
