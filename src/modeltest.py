from lstmmodel import LSTMModel
from nnmodel import NNModel
from model import Model
import matplotlib.pyplot as plt
from modelplotter import ModelPlotter
from regressionmodel import RegressionModel
from csvList import trainPaths, validationPaths, sensVsDist
from dataset import Dataset

TIMESTEPS = 10
VALIDATIONRATIO = 0.3
EPOCHS = 300
print("Hello World")
nnmodel = NNModel()
lstmmodel = LSTMModel()
lstmmodel.initModel(timesteps=TIMESTEPS)
regressionmodel = RegressionModel()

dataset = Dataset.fromCSVList(sensVsDist, timesteps=TIMESTEPS)
# models = [regressionmodel]
# models = [nnmodel, lstmmodel, regressionmodel]
# models = [nnmodel, lstmmodel]
models = [lstmmodel]
preductionsAndTests = []
for model in models:
    history = model.trainOnDataset(dataset, VALIDATIONRATIO, epochs=EPOCHS)
    mae, yPred, yTrue = model.predictOnDataset(dataset, VALIDATIONRATIO)
    print(f"Model Finished! Validation mae: {mae}")
    # if model == lstmmodel:
    ModelPlotter.plotLearningCurves(history, model.scaler)
    preductionsAndTests.append((model.modelName, yPred, yTrue))
ModelPlotter.predVsTrue(preductionsAndTests)
# for path in trainPaths:
#     model.trainCSV(path)

# print("Ready to predict")
# input()


for path in validationPaths:
    fig1 = ModelPlotter.plotEstimateOnCSV(models, path)
    plt.show()

# model.plot()
