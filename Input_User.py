import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import seaborn as sns

# Function to calculate Euclidean distance
def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

def train_naive_bayes_models(dataset):
    # Define feature columns (WAP data) and target columns (FLOOR, BUILDINGID)
    X = dataset.iloc[:, :520]  # WAP columns
    y_floor = dataset['FLOOR']  # Target: FLOOR classification
    y_building = dataset['BUILDINGID']  # Target: BUILDING classification

    # Split data into training and testing sets
    X_train, X_test, y_train_floor, y_test_floor, y_train_building, y_test_building = train_test_split(
        X, y_floor, y_building, test_size=0.3, random_state=42)

    # Initialize the Bayes model
    bayes_model_floor = GaussianNB()
    bayes_model_building = GaussianNB()

    # Train the model on FLOOR classification
    bayes_model_floor.fit(X_train, y_train_floor)

    # Train the model on BUILDING classification
    bayes_model_building.fit(X_train, y_train_building)

    return bayes_model_floor, bayes_model_building

def animate_user_movement(user_input_file, user_id, bayes_model_floor, bayes_model_building):
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
    timestamps = user_data['TIMESTAMP'].values

    # Prepare data for prediction
    X = user_data.iloc[:, :520].values
    # Predict FLOOR and BUILDINGID using the Naive Bayes models
    pred_floors = bayes_model_floor.predict(X)
    pred_buildings = bayes_model_building.predict(X)

    # Create figure and axis
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(color="lightgreen")

    ax.set_xlim(longitudes.min() - 0.01, longitudes.max() + 0.01)
    ax.set_ylim(latitudes.min() - 0.01, latitudes.max() + 0.01)
    ax.set_title(f"User Movement for User ID {user_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Plot initial points
    actual_scatter = ax.scatter(longitudes[0], latitudes[0], c='blue', s=100)
    pred_scatter = ax.scatter(longitudes[0], latitudes[0], c='red', s=100)  # Initial prediction is the same as actual

    actual_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='blue')
    predicted_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='red')

    def update(frame):
        if frame >= len(longitudes):
            return actual_scatter, pred_scatter, actual_text, predicted_text

        actual_scatter.set_offsets([longitudes[frame], latitudes[frame]])
        pred_scatter.set_offsets([longitudes[frame], latitudes[frame]])  # Placeholder for predicted lat/lon

        actual_text.set_text(f"Actual Longitude: {round(longitudes[frame],2)} || Actual Latitude: {round(latitudes[frame],2)}")
        predicted_text.set_text(f"Predicted Floor: {pred_floors[frame]} || Predicted Building: {pred_buildings[frame]}")
        return actual_scatter, pred_scatter, actual_text, predicted_text

    ani = animation.FuncAnimation(fig, update, frames=len(longitudes), interval= 50, blit=True, repeat=False)
    plt.show()

# Load your dataset
dataset = pd.read_csv(r'C:\Users\Hp\OneDrive\Indoor Localization using RSSI Dataset\archive (1)\TrainingData.csv')

# Train the Naive Bayes models
bayes_model_floor, bayes_model_building = train_naive_bayes_models(dataset)

# Example usage
user_input_file = r'C:\Users\Hp\OneDrive\Indoor Localization using RSSI Dataset\archive (1)\TrainingData.csv'
user_id = int(input("Enter the User ID: "))
animate_user_movement(user_input_file, user_id, bayes_model_floor, bayes_model_building)

