import tensorflow as tf
import numpy as np
import os
import keras 

try:
  model = keras.models.load_model("my_model.h5")
except:
  model = keras.Sequential([keras.layers.Dense(units=1 , input_shape=[1])])
  model.compile(optimizer="sgd" , loss="mean_squared_error")
  x = np.array([-1.0 , 0.0 , 1.0 , 2.0 , 3.0 , 4.0] , dtype="float")
  y = np.array([-3.0 , -1.0 , 1.0 , 3.0 , 5.0 , 7.0] , dtype="float")
  model.fit(x , y , epochs=500)


model.save('my_model.h5')

print(model.predict([20]))
print(model.predict([7]))
print(model.predict([4]))
print(model.predict([1000]))
print(model.predict([50]))
print(model.predict([10]))
