from bayes_opt import BayesianOptimization
import time
from lstmmodel import LSTMModel
from nnmodel import NNModel
from model import Model
import matplotlib.pyplot as plt
from modelplotter import ModelPlotter
from regressionmodel import RegressionModel
import tensorflow.keras
import numpy as np
import tensorflow as tf
from csvList import trainPaths, validationPaths, sensVsDist
from dataset import Dataset
import datetime
from scaler import Scaler

TIMESTEPS = 10
VALIDATIONRATIO = 0.3
EPOCHS = 1000
# POINTS_TO_EVALUATE = 20
POINTS_TO_EVALUATE = 50

# current date

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Hello World")
dataset = Dataset.fromCSVList(sensVsDist, timesteps=TIMESTEPS)

with open(f"bayes_optimization_log{now}.txt", "a") as f:
    f.write(f"Features: {Dataset.acceptedFeatures}\n")


def evaluate_network(
    dropout, learning_rate, neuronPct, neuronShrink, kernel_regularizer
):
    start_time = time.time()
    # Below are a list of csv paths inside a list variable we use to train from multiple csv files

    print("Hello World")
    lstmmodel = LSTMModel()
    lstmmodel.initModel(
        TIMESTEPS, learning_rate, neuronPct, neuronShrink, kernel_regularizer, dropout
    )

    params = [dropout, learning_rate, neuronPct, neuronShrink, kernel_regularizer]

    print("===========(BaysianOP)Evaluating network with parameters:")
    print(f"dropout: {dropout}")
    print(f"learning_rate: {learning_rate}")
    print(f"neuronPct: {neuronPct}")
    print(f"neuronShrink: {neuronShrink}")
    print(f"timesteps: {TIMESTEPS}")
    print(f"kernel_regularizer: {kernel_regularizer}")
    print("===========(BaysianOP)Training on dataset")
    # lstmmodel.print_dataset_info("Training dataset", training_dataset)
    # lstmmodel.print_dataset_info("validation dataset", validation_dataset)

    history = lstmmodel.trainOnDataset(dataset, VALIDATIONRATIO, epochs=EPOCHS)
    print("===========(BaysianOP)Training complete, Predicting on validation dataset")
    mae, _, _ = lstmmodel.predictOnDataset(dataset, VALIDATIONRATIO)
    scaler = lstmmodel.getScaler().getScaler("temperature")
    scale_factor = scaler.data_max_ - scaler.data_min_
    historyOriginalValMAE = history.history["val_mae"] * scale_factor
    last_50_val_mae = historyOriginalValMAE[-50:]
    avg_val_mae = np.mean(last_50_val_mae)
    print(f"===========(BaysianOP)Validation mae: {mae} average: {avg_val_mae}")
    # Record this iteration
    time_took = time.time() - start_time
    print(
        f"===========(BaysianOP)Time took for iteration with error of {mae}: {time_took}"
    )

    with open(f"bayes_optimization_log{now}.txt", "a") as f:
        f.write(f"Params: {params}, Error: {-avg_val_mae:.6f}\n")

    tensorflow.keras.backend.clear_session()
    return -avg_val_mae


def hms_string(sec_elapsed):
    h = int(sec_elapsed // 3600)
    m = int((sec_elapsed % 3600) // 60)
    s = int(sec_elapsed % 60)
    return f"{h}h {m}m {s}s"


# Bounded region of parameter space
# pbounds = {
#     "dropout": (0.0, 0.1),
#     "learning_rate": (0.0001, 0.01),
#     "neuronPct": (0.001, 1),
#     "neuronShrink": (0.2, 1),
#     "timesteps": (1, 10),
#     "kernel_regularizer": (0.0001, 0.01),
# }
pbounds = {
    "dropout": (0.0, 0.002),
    "learning_rate": (0.0009, 0.0011),
    "neuronPct": (0.005, 0.10),
    "neuronShrink": (0.4, 1),
    "kernel_regularizer": (0.0009, 0.0011),
}


optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum
    # is observed, verbose = 0 is silent
    random_state=1,
)

start_time = time.time()
optimizer.maximize(
    init_points=POINTS_TO_EVALUATE,
    n_iter=25,
)
time_took = time.time() - start_time

with open(f"bayes_optimization_log{now}.txt", "a") as f:
    f.write(
        f"Params: {optimizer.max['params']}, Error: {optimizer.max['target']:.6f}\n"
    )
    f.write(f"Runtime: {hms_string(time_took)}")

print(optimizer.max)
