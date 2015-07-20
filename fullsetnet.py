import numpy
import time
from sklearn import preprocessing
from sklearn import metrics
from sknn.mlp import Classifier, Regressor, Layer
training_data = "FullModemConfigSpace-2015-07-19.csv"
training = numpy.loadtxt(open(training_data), delimiter=",", skiprows = 1)
testing_data = "path to dataset used for testing here"
testing = numpy.loadtxt(open(testing_data), delimiter="space, comma, semicolon, whatever separates the attributes in the samples")
"""skiprows=1 if the attribute names are at the top"""
tr_x = dataset[:,0:38]
tr_y = training[:,39:43]
ts_x = testing[:,0:38]
ts_y = testing[:,39:43]
network = Regressor(layers=[Layer("Linear",units=39), Layer("Sigmoid",units=22), Layer("Linear", units=4)], learning_rate=0.001, n_iter=25)
network.fit(tr_x,tr_y)
cont_prediction = cont_network.predict(ts_x)
"""This will write the predictions to an output file, I'm pretty sure you can change the extension as you like."""
with open("full_prediction.txt", "w") as output:
	for y in cont_prediction:
		print(y, file=output)