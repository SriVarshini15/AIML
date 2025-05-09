'''
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)
'''


from numpy import loadtxt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = loadtxt('/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning2/Day_11/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]
print(x)


model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Model Training
model.fit(x, y, epochs=40, batch_size=10)

#Evaluation
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

#Model Save
model_json = model.to_json()
with open("/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning2/Day_11/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning2/Day_11/model.h5")
print("Saved model to disk")
