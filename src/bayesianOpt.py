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


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Hello World")


def evaluate_network(
    dropout, learning_rate, neuronPct, neuronShrink, timesteps, kernel_regularizer
):
    start_time = time.time()
    # Below are a list of csv paths inside a list variable we use to train from multiple csv files
    csvList = [
        "./data/run-20240815_152530/csv_20240815_152530.csv",
        "./data/run-20240815_145106/csv_20240815_145106.csv",
        "./data/run-20240712_160241/csv_20240712_160241.csv",
        "./data/run-20240710_154651/csv_20240710_154651.csv",
    ]

    print("Hello World")
    nnmodel = NNModel()
    lstmmodel = LSTMModel()
    lstmmodel.initModel(
        timesteps, learning_rate, neuronPct, neuronShrink, kernel_regularizer, dropout
    )
    combinedDataset = lstmmodel.datasetFromCSVList(csvList)
    training_dataset, validation_dataset = lstmmodel.splitDataset(combinedDataset)
    print("===========(BaysianOP)Evaluating network with parameters:")
    print(f"dropout: {dropout}")
    print(f"learning_rate: {learning_rate}")
    print(f"neuronPct: {neuronPct}")
    print(f"neuronShrink: {neuronShrink}")
    print(f"timesteps: {timesteps}")
    print(f"kernel_regularizer: {kernel_regularizer}")
    print("===========(BaysianOP)Training on dataset")
    lstmmodel.print_dataset_info("Training dataset", training_dataset)
    lstmmodel.print_dataset_info("validation dataset", validation_dataset)

    history = lstmmodel.trainOnDataset(training_dataset, validation_dataset)
    print("===========(BaysianOP)Training complete, Predicting on validation dataset")
    mae = lstmmodel.predictOnDataset(validation_dataset)
    print(f"===========(BaysianOP)Validation mae: {mae}")
    # Record this iteration
    time_took = time.time() - start_time
    print(
        f"===========(BaysianOP)Time took for iteration with error of {mae}: {time_took}"
    )

    tensorflow.keras.backend.clear_session()
    return -mae


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
    "dropout": (0.0, 0.001),
    "learning_rate": (0.0009, 0.0011),
    "neuronPct": (0.005, 0.015),
    "neuronShrink": (0.8, 1),
    "timesteps": (6, 10),
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
    init_points=10,
    n_iter=20,
)
time_took = time.time() - start_time

print(f"Total runtime: {hms_string(time_took)}")
print(optimizer.max)
