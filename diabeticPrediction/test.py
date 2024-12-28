from numpy import loadtxt
from keras.models import model_from_json
dataset = loadtxt('/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning2/Day_11/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning2/Day_11/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning2/Day_11/model.h5")
print("Loaded model from disk")

predictions = model.predict(x)
for i in range(5,10):
	print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
