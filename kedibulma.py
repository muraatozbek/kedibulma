
# kaynak 
'''
http://www.cs.toronto.edu/~kriz/cifar.html
'''



from scipy import ndimage
from scipy import misc
import numpy
from matplotlib import pyplot
from scipy.misc import toimage
import scipy.misc

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import keras.layers
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')




(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# normalize inputs from 0-255 to 0.0-1.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

for i in range(0, 9):
 pyplot.subplot(330 + 1 + i)
 pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()


kedi = ndimage.imread("kedi.jpg")
kedi = scipy.misc.imresize(kedi,(32,32))
kedi = numpy.array(kedi)
print("jfjef")
kedi = kedi.reshape(1,3,32,32)

model = Sequential()
#giriso larak 3 katmanli 32*32 lik bir foto aldik
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#daha sonra bunu convolution yaparak 3*3 luk matris ile 32 lik pikseli (3-1)=2 ,32-2=30 piksele indirdik
model.add(MaxPooling2D(pool_size=(2, 2)))
#daha sonra pooling ile 30 u 2*2 yaptigimiz icin 15 e indridik 30/2=15
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#tekrar con yaptik bu seferde 3*3 luk yani pikselden 2 eksilticez 13 oldu
model.add(MaxPooling2D(pool_size=(2, 2)))
#tekrar pooling yaparak 2 ye bolucez 13/2 = 6.5
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#tekrar conv yaptik 2 eksilticez 6.5-2=4.5
model.add(MaxPooling2D(pool_size=(2, 2)))
#tekrar pooling yaptik 2 ye bolucez 4.5/2=2.25
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#tekrar conv yaptik 2 eksilticez 2.25-2=0.25
model.add(MaxPooling2D(pool_size=(2, 2)))
#tekrar pooling yaptik 2 ye bolucez 0.25/2=0.125

model.add(Flatten())
#elimizde kalan pikselleri duzlestirdik tek satir haline getirdik
model.add(Dense(1000, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
#kodun ezberlememesi icn dropout yaptik 0.5 ihtimalle bazi hucreleri es gecicek
model.add(Dense(1000, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Compile model
epochs = 300
lrate = 0.001
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print("---------------")

print(model.predict_classes(kedi))