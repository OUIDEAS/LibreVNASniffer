from lstmmodel import LSTMModel
from nnmodel import NNModel
from model import Model
import matplotlib.pyplot as plt
from modelplotter import ModelPlotter
from regressionmodel import RegressionModel
from csvList import trainPaths, validationPaths, sensVsDist
from dataset import Dataset

TIMESTEPS = 20
VALIDATIONRATIO = 0.3
EPOCHS = 1000
print("Hello World")
nnmodel = NNModel()
lstmmodel = LSTMModel()
lstmmodel.initModel(timesteps=TIMESTEPS)
regressionmodel = RegressionModel()

dataset = Dataset.fromCSVList(sensVsDist, timesteps=TIMESTEPS)
# models = [regressionmodel]
models = [lstmmodel, nnmodel, regressionmodel]
# models = [nnmodel, lstmmodel]
# models = [lstmmodel]
# models = [regressionmodel, lstmmodel]
preductionsAndTests = []
for model in models:
    history = model.trainOnDataset(dataset, VALIDATIONRATIO, epochs=EPOCHS)
    mae, yPred, yTrue = model.predictOnDataset(dataset, VALIDATIONRATIO)
    print(f"Model " + model.modelName + f"Finished! Validation mae: {mae}")
    # if model == lstmmodel:
    ModelPlotter.plotLearningCurves(history, model.scaler, model.modelName)
    preductionsAndTests.append((model.modelName, yPred, yTrue))
ModelPlotter.predVsTrue(preductionsAndTests)
# for path in trainPaths:
#     model.trainCSV(path)

# print("Ready to predict")
# input()


for path in validationPaths:
    fig1 = ModelPlotter.plotEstimateOnCSV(models, path)
    # plt.show()

# model.plot()
