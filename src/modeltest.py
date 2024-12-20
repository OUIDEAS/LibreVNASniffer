from lstmmodel import LSTMModel
from nnmodel import NNModel
from model import Model
import matplotlib.pyplot as plt
from modelplotter import ModelPlotter
from regressionmodel import RegressionModel

# Below are a list of csv paths inside a list variable we use to train from multiple csv files
trainPaths = [
    "./data/run-20240815_152530/csv_20240815_152530.csv",
    "./data/run-20240815_145106/csv_20240815_145106.csv",
    "./data/run-20240712_160241/csv_20240712_160241.csv",
    "./data/run-20240710_154651/csv_20240710_154651.csv",
]


print("Hello World")
nnmodel = NNModel()
lstmmodel = LSTMModel(10)
regressionmodel = RegressionModel()
# models = [regressionmodel]
models = [nnmodel, lstmmodel, regressionmodel]
for model in models:
    history = model.miniBatchTrain(trainPaths)
    # model.plot_learning_curves(history)

# for path in trainPaths:
#     model.trainCSV(path)

# print("Ready to predict")
# input()


for path in trainPaths:
    fig1 = ModelPlotter.plotEstimateOnCSV(models, path)
    plt.show()

# model.plot()
