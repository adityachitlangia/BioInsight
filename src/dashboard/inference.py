# inference.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd

# Configuration
LOOKBACK = 60
FEATURES = ['kLa (1/h)', 'DO (%)', 'pH']
MODEL_PATH = '/Users/nrcase/BioInsight/src/dashboard/models/bioreactor_model.h5'
SCALER_PATH = '/Users/nrcase/BioInsight/src/dashboard/models/robust_scaler.save'


def forecast_with_uncertainty(model, initial_sequence, steps, n_samples=100):
    predictions = []
    current_seq = initial_sequence.copy()

    for _ in range(steps):
        # Enable dropout for uncertainty estimation
        for layer in model.layers:
            if 'dropout' in layer.name.lower():
                layer.trainable = True

        # Generate multiple predictions
        preds = np.array([model.predict(current_seq[np.newaxis, ...])[0]
                          for _ in range(n_samples)])

        # Disable dropout
        for layer in model.layers:
            if 'dropout' in layer.name.lower():
                layer.trainable = False

        # Update sequence
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = preds.mean(axis=0)

        predictions.append(preds)

    return np.array(predictions)


def run_inference(data):
    # Load artifacts
    # Load model with MSE explicitly defined
    model = load_model(
        MODEL_PATH,
        custom_objects={'mse': MeanSquaredError()}
    )

    scaler = joblib.load(SCALER_PATH)
    test_data = data
    print(data)

    # Prepare test sequence
    test_sequence = test_data[FEATURES].values[:LOOKBACK]
    scaled_sequence = scaler.transform(test_sequence)

    # Generate forecasts
    predictions = forecast_with_uncertainty(model, scaled_sequence, steps=20)
    predictions_unscaled = scaler.inverse_transform(
        predictions.reshape(-1, 3)).reshape(predictions.shape)

    # Create plot
    plt.figure(figsize=(15, 10))
    time_historical = np.arange(LOOKBACK)
    time_forecast = np.arange(LOOKBACK, LOOKBACK+20)

    for i, feature in enumerate(FEATURES):
        plt.subplot(3, 1, i+1)

        # Historical data
        plt.plot(time_historical,
                 test_sequence[:, i], 'b-', label='Historical')

        # Forecasts
        mean_pred = predictions_unscaled[:, :, i].mean(axis=1)
        std_pred = predictions_unscaled[:, :, i].std(axis=1)

        plt.plot(time_forecast, mean_pred, 'r--', label='Forecast')
        plt.fill_between(time_forecast,
                         mean_pred - 1.96*std_pred,
                         mean_pred + 1.96*std_pred,
                         color='r', alpha=0.2, label='95% CI')

        plt.title(f'{feature} Forecast')
        plt.xlabel('Time Steps')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('forecast_plot.png')


if __name__ == "__main__":
    run_inference()
