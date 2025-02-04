from lstmmodel import LSTMModel
from nnmodel import NNModel
from model import Model
import matplotlib.pyplot as plt
from modelplotter import ModelPlotter
from regressionmodel import RegressionModel
from csvList import trainPaths, validationPaths


print("Hello World")
nnmodel = NNModel()
lstmmodel = LSTMModel(10)
lstmmodel.initModel()
regressionmodel = RegressionModel()
# models = [regressionmodel]
# models = [nnmodel, lstmmodel, regressionmodel]
models = [lstmmodel]
for model in models:
    (history, mae) = model.miniBatchTrain(trainPaths)
    print(f"Model Finished! Validation mae: {mae}")
    if model == lstmmodel:
        model.plot_learning_curves(history)
# for path in trainPaths:
#     model.trainCSV(path)

# print("Ready to predict")
# input()


for path in validationPaths:
    fig1 = ModelPlotter.plotEstimateOnCSV(models, path)
    plt.show()

# model.plot()
