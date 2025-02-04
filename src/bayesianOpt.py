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
from csvList import trainPaths, validationPaths

TIMESTEPS = 10

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Hello World")
lstmmodel = LSTMModel(TIMESTEPS)
lstmmodel.initModel()
combinedDataset = lstmmodel.datasetFromCSVList(trainPaths)
training_dataset, validation_dataset = lstmmodel.splitDataset(combinedDataset)
# get scalers
xScale, yScale = lstmmodel.getScalers()


def evaluate_network(
    dropout, learning_rate, neuronPct, neuronShrink, kernel_regularizer
):
    start_time = time.time()
    # Below are a list of csv paths inside a list variable we use to train from multiple csv files

    print("Hello World")
    nnmodel = NNModel()
    lstmmodel = LSTMModel()
    lstmmodel.initModel(
        TIMESTEPS, learning_rate, neuronPct, neuronShrink, kernel_regularizer, dropout
    )
    lstmmodel.setScalers(xScale, yScale)

    print("===========(BaysianOP)Evaluating network with parameters:")
    print(f"dropout: {dropout}")
    print(f"learning_rate: {learning_rate}")
    print(f"neuronPct: {neuronPct}")
    print(f"neuronShrink: {neuronShrink}")
    print(f"timesteps: {TIMESTEPS}")
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
    init_points=20,
    n_iter=25,
)
time_took = time.time() - start_time

print(f"Total runtime: {hms_string(time_took)}")
print(optimizer.max)
