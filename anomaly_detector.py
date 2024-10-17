import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report

"""
this code simulates a data stream with anomalies and uses a sliding window to detect anomalies in real-time.
it uses the Local Outlier Factor (LOF) algorithm to detect anomalies.
The code initializes a visualization using matplotlib and iterates over the data stream, updating the visualization and detecting anomalies at each time step.
The code also calculates and prints accuracy metrics using the classification_report function from scikit-learn.
data simulation: The code simulates a data stream with a linear baseline, a sine seasonal component, and noise.
there is a warmup period during which the model learns the normal pattern before detecting anomalies for better accuracy

changing components of this code is simple and can be done by changing the variables defines in main function only
including the model used for anomaly detection, the baseline and seasonal functions, the noise standard deviation, the anomaly indices, and the anomaly magnitude.
"""

def simulate_data_stream(length, baseline_func, seasonal_func, noise_std, anomaly_indices, anomaly_magnitude, warmup_period):
    time = np.arange(length)
    baseline = baseline_func(time) # main axis of signal
    seasonal = seasonal_func(time) # shapes the signal with function
    noise = np.random.normal(0, noise_std, length) #adds noise to the signal
    data_stream = baseline + seasonal + noise # final signal
    labels = np.ones(length)  # Initialize all as normal
     # Ensure anomalies are only added after the warm-up period
    valid_anomaly_indices = [idx for idx in anomaly_indices if idx >= warmup_period]
    data_stream[valid_anomaly_indices] += anomaly_magnitude  # Add anomalies
    labels[valid_anomaly_indices] = -1  # Set anomalies

    for data_point, label in zip(data_stream, labels):
        yield data_point, label # Return data point and label

def initialize_visualization():
    plt.ion()  # Turn on interactive mode for real-time plotting
    ax = plt.subplots()  # Create figure and axis objects
    line, = ax.plot([], [], label='Data stream')  # Plot for data stream
    anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')  # Plot for anomalies
    ax.legend()  # Display legend
    plt.show()  # Display the plot

    data_points = []
    anomaly_indices = []
    anomaly_values = []

    def update_visualization(data_point, anomaly_status): # Update the visualization with parameters passed in initialize_visualization
        data_points.append(data_point)
        line.set_data(np.arange(len(data_points)), data_points) # Update the data stream plot

        if anomaly_status == -1:
            anomaly_indices.append(len(data_points) - 1) # Add the index of the anomaly
            anomaly_values.append(data_point) # Add the value of the anomaly
        anomaly_points.set_data(anomaly_indices, anomaly_values) # Update the anomaly plot

        ax.relim() # Update limits of the plot
        ax.autoscale_view() # Update view
        plt.draw() # Redraw the plot
        plt.pause(0.01) # Pause to allow for smooth animation

    return update_visualization

def detect_anomalies(data_point, model, window, window_size, warmup_period, current_index):
    window.append(data_point) # Append the new data point to the window
    if len(window) > window_size: # If the window is full, remove the oldest data point
        window.pop(0)
    if current_index < warmup_period: 
        return data_point, 1  # Assume normal during warm-up
    if len(window) == window_size: # If the window is full, fit the model and predict
        model.fit(np.array(window).reshape(-1, 1))
        anomaly_status = model.predict([[data_point]])[0]
        return data_point, anomaly_status
    return data_point, 1  # Default to normal if window is not yet full

def main(): 
    # Define parameters for data simulation
    length = 1000
    window_size = 300
    noise_std = 0.5
    baseline_func = lambda t: 0.01 * t  # Linear baseline
    seasonal_func = lambda t: 10 * np.sin(2 * np.pi * t / 50)  # Sine seasonal component
    anomaly_indices = np.random.choice(length, size=int(0.03 * length), replace=False)  # Random anomalies
    anomaly_magnitude = 20  # Magnitude of the anomalies
    warmup_period = 200 # Warmup period for the model to recognize the normal pattern
    model = LocalOutlierFactor(n_neighbors=50, contamination=0.025, novelty=True)  # LOF model

    data_stream = simulate_data_stream(length, baseline_func, seasonal_func, noise_std, anomaly_indices, anomaly_magnitude, warmup_period)
    #model = IsolationForest(contamination=0.01)  # 1% of data points are anomalies
    
    window = []  # Initialize the window for storing recent data points
    update_visualization = initialize_visualization()

    true_labels = []
    predicted_labels = []

    for data_point, true_label in data_stream:
        data_point, anomaly_status = detect_anomalies(data_point, model, window, window_size, warmup_period, len(true_labels)) # Detect anomalies
        update_visualization(data_point, anomaly_status) # Update the visualization
        print("Data point: {}, Anomaly status: {}".format(data_point, anomaly_status))

        true_labels.append(true_label)
        predicted_labels.append(anomaly_status)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot window open

    # Calculate and print accuracy metrics
    print(classification_report(true_labels[warmup_period:], predicted_labels[warmup_period:], target_names=['Normal', 'Anomaly']))

if __name__ == "__main__":
    main()