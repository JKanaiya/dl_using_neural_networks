"""
=======================================================================================
Python script to demonstrate corp yield prediction using shallow and deep networks
=======================================================================================

This program demonstrates maize yield production using Shallow and Deep Networks using a synthetic maize yield dataset
The simulated dataset simulates the relationship between soil nutrients, weather conditions

Requirements:
    matplotlib numpy pandas seaborn scikit-learn tensorflow keras

"""

# -----------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Constants and Global Configuration
# -----------------------------------------------------------------------------

DATA_FILENAME = "files/african_maize_yield_data.csv"
RANDOM_SEED: int = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# 2. Dataset Generation
# -----------------------------------------------------------------------------
def generate_dataset(num_samples: int = 5000) -> pd.DataFrame:
    # Fertiliser constitution
    nitrogen = np.random.uniform(20, 120, num_samples)
    phosphorus = np.random.uniform(10, 60, num_samples)
    potassium = np.random.uniform(30, 150, num_samples)

    # Weather
    rainfall = np.random.uniform(400, 1400, num_samples)
    temperature = np.random.uniform(10, 25, num_samples)

    # soil organic percentage
    organic_matter = np.random.uniform(1.0, 5.0, num_samples)

    # -------------------------------------------------------------------------
    # Simulated yield model
    # -------------------------------------------------------------------------
    linear_yield = 0.1 * nitrogen + 0.15 * phosphorus + 0.05 * potassium

    drought_factor = np.where(
        (rainfall < 600) & (temperature > 30), -14 * (36 - temperature), 0
    )

    buffer_factor = organic_matter * (rainfall / 200)
    temperature_efficiency = 20 * np.sin((temperature - 18) / 18 * np.pi)

    # Add random noise to make the dataset more realistic
    noise = np.random.normal(0, 2, num_samples)

    # maize yield
    yield_bags = (
        15
        + linear_yield
        + drought_factor
        + buffer_factor
        + temperature_efficiency
        + noise
    )
    yield_bags = np.clip(yield_bags, 2, 90)

    # genearte dataset to be returned
    dataset = pd.DataFrame(
        {
            "Nitorgen_NPK": nitrogen,
            "Phosphor_NPK": phosphorus,
            "Potassium": potassium,
            "Average_rainfall_mm": rainfall,
            "Average_temperature_C": temperature,
            "Soil_Organic_Matter_Pct": organic_matter,
            "Maize_Yield_Bags_Per_Ha": yield_bags,
        }
    )

    return dataset


# -----------------------------------------------------------------------------
# 3. Dataset storage
# -----------------------------------------------------------------------------
def save_dataset(dataset: pd.DataFrame, filename: str = DATA_FILENAME) -> None:
    dataset.to_csv(filename, index=False)
    print(f"\nDataset successfully saved as: \n{filename}")


def load_dataset(filename: str = DATA_FILENAME) -> pd.DataFrame:
    if not Path(filename).exists():
        raise FileNotFoundError(f"Dataset file '{filename}' does not exist")
    print(f"\nDataset successfully loaded from '{filename}'")

    return pd.read_csv(filename)


# -----------------------------------------------------------------------------
# 4. Data Explanation
# -----------------------------------------------------------------------------
def display_dataset_information(dataset: pd.DataFrame) -> None:
    print("-" * 70)
    print("DATASET OVERVIEW")
    print("-" * 70)

    print(dataset.head())

    print(f"\nDataset Shape: \n{dataset.shape}")
    print(f"\nSummary Statistics: \n{dataset.describe()}")
    print(f"\nMissing Values: \n{dataset.isnull().sum()}")


# -----------------------------------------------------------------------------
# 5. Correlaiton Heatmap
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(dataset: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap="viridis", fmt=".2f")
    plt.title(
        "Correlaiton Matrix of the African Maize Yield Dataset",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 6. Data Preprocessing
# -----------------------------------------------------------------------------
def process_data(
    dataset: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:

    print("\nPreprocessing Dataset")
    X = dataset.drop(columns=["Maize_Yield_Bags_Per_Ha"]).values

    y = dataset["Maize_Yield_Bags_Per_Ha"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Dataset successfully preprocessed")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    return (X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler)


# -----------------------------------------------------------------------------
# 7. Shallow Network
# -----------------------------------------------------------------------------
def build_shallow_model(input_dimension: int) -> Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dimension,)),
            layers.Dense(8, activation="relu", name="Hidden_Layer_1"),
            layers.Dense(1, activation="linear", name="Output"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# -----------------------------------------------------------------------------
# 8. Deep Network
# -----------------------------------------------------------------------------
def build_deep_model(input_dimension: int) -> Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dimension,)),
            layers.Dense(128, activation="relu", name="Hidden_Layer_1"),
            layers.Dense(64, activation="relu", name="Hidden_Layer_2"),
            layers.Dense(32, activation="relu", name="Hidden_Layer_3"),
            layers.Dense(16, activation="relu", name="Hidden_Layer_4"),
            layers.Dense(1, activation="linear", name="Output"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# -----------------------------------------------------------------------------
# 9. Model Visualisation
# -----------------------------------------------------------------------------
def display_model_summary(model: Model, model_name: str) -> None:
    print("-" * 70)
    print(f"{model_name.upper()}")
    print("-" * 70)

    model.summary()


def save_model_architecture(model: Model, filename: str) -> None:
    try:
        plot_model(
            model, to_file=filename, show_shapes=True, show_layer_names=True, dpi=150
        )
    except Exception as err:
        print(f"\nArchitecture diagram could not be generated.\n{err}")


# -----------------------------------------------------------------------------
# 10. Early Stopping
# -----------------------------------------------------------------------------


def create_early_stopping() -> EarlyStopping:
    return EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )


# -----------------------------------------------------------------------------
# 11. Model Training
# -----------------------------------------------------------------------------
def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epoch: int = 100,
    batch_size: int = 32,
) -> tf.keras.callbacks.History:
    callback = create_early_stopping()

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=1,
    )

    return history


# -----------------------------------------------------------------------------
# 12. Save Model
# -----------------------------------------------------------------------------
def save_trained_model(model: Model, filename: str) -> None:
    model.save(filename)
    print(f"\nModel saved to:\n '{filename}'")


# -----------------------------------------------------------------------------
# 13. Model Evaluation
# -----------------------------------------------------------------------------
def evaluate_model(
    model: Model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> dict:
    predictions = model.predict(X_test, verbose=0).flatten()

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} Overview")
    print("=" * 70)

    print(
        f"Mean Squared Error           :{mse:.3f}"
        f"\nRoot Mean Squared Error    :{rmse:.3f}"
        f"\nMean Absolute Error        :{rmse:.3f}"
        f"\nR Score                    :{rmse:.3f}"
    )
    return {
        "name": model_name,
        "predictions": predictions,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# -----------------------------------------------------------------------------
# 14. Loss Curves
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# n. Main execution function
# -----------------------------------------------------------------------------
def main() -> None:
    pass


# -----------------------------------------------------------------------------
# n. Run the script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
