## Real-time Anomaly Detection in Data Streams

This repository contains Python code for simulating a data stream with anomalies and detecting them in real-time using a sliding window approach. The code utilizes the Local Outlier Factor (LOF) algorithm for anomaly detection and visualizes the results.

### Installation

There are no specific dependencies beyond Python and common scientific libraries. You can install the required libraries using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

1. Clone repository
```bash
git clone https://github.com/mariam2001/Anomaly-Detector.git
```
2. Run the script
```bash
python main.py
```

This will generate a simulated data stream, detect anomalies using the LOF model, and display a real-time visualization of the results. The script will also print a classification report for evaluation after the warmup period.

## Code Structure

  * `simulate_data_stream.py`: Defines functions to simulate a data stream with anomalies.
  * `initialize_visualization.py`: Initializes the real-time visualization using matplotlib.
  * `detect_anomalies.py`: Implements the sliding window approach and anomaly detection using the LOF model.
  * `main.py`: The main script that configures parameters, runs the simulation, and displays the results.

### Customization

The code allows customization of various parameters within the `main.py` script. These include:

  * `length`: Length of the data stream.
  * `window_size`: Size of the sliding window for anomaly detection.
  * `noise_std`: Standard deviation of the noise added to the data stream.
  * `baseline_func`: Function defining the baseline trend of the data.
  * `seasonal_func`: Function defining the seasonal component of the data.
  * `anomaly_indices`: List of indices where anomalies are introduced.
  * `anomaly_magnitude`: Magnitude of the anomaly deviations.
  * `warmup_period`: Number of initial data points used for model training.
  * `model`: The anomaly detection model (currently uses LOF, can be replaced with IsolationForest)

I welcome contributions from the community to further refine and improve this code.
