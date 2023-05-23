#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras_tuner
from tensorflow import keras
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers
from tensorflow.keras import optimizers


# # Preprocessing

#     The pictures are already divided into train, valid, and test sets, so there is no need to make further adjustments on the disk.

# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


train_path = '/Users/ianwallgren/Desktop/UPC/Courses/NeuralNetworks/Assignments/Assignment_2/BigCats10/train'
valid_path = '/Users/ianwallgren/Desktop/UPC/Courses/NeuralNetworks/Assignments/Assignment_2/BigCats10/valid'
test_path = '/Users/ianwallgren/Desktop/UPC/Courses/NeuralNetworks/Assignments/Assignment_2/BigCats10/test'


#theese are the classes we have on disc
classes = ['AFRICAN LEOPARD','CARACAL','CHEETAH','CLOUDED LEOPARD','JAGUAR','LIONS','OCELOT',          'PUMA','SNOW LEOPARD','TIGER']

#pictures will be resized to 224 by 224 pixels
target_size = (224,224)

#not sure if this choice will affect the performance of the model very much (experiment with this).
batch_size = 10


# In[4]:


#applying the same preprocessing function to all the images of the different batches (train,valid,test) as the 
#vgg16 CNN. Note that the way of preprocessing the data will have a direct impact on the performance,
#so we keep in mind that this could be a potential source of error if our model displays poor performance metrics.

#the chunk below essentially creates batches of data from our directiories where the train, valid and test sets
#reside, and these batches will then be able to be past to our models when we train/predict at a later step.
train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).                flow_from_directory(directory = train_path, target_size = target_size, classes = classes,                batch_size = batch_size)

valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).                flow_from_directory(directory = valid_path, target_size = target_size, classes = classes,                batch_size = batch_size)

#we want to set shuffle = False for the test batches in order to be able to generate a valid confusion matrix later
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).                flow_from_directory(directory = test_path, target_size = target_size, classes = classes,                batch_size = batch_size, shuffle = False)


#     We see that we have 2339 images in total for the training set, and these are divided into the 10 different classes. We then have 50 and 50 images belonging to the valid and test sets, respectively.

# In[5]:


#getting a batch of images and corresponding labels from hte train batches, with the purpose of making sure we
#have successfully created our batches in the above chunk.
images, labels = next(train_batches)


# In[6]:


#function below is directly taken from Keras with some necessary alterations

def plotImages(images_arr,lbls):
    #we plot 10 subplots for each picture in our batch (which is 10 according to the current batch size)
    fig, axes = plt.subplots(1, batch_size, figsize = (20,20))
    axes = axes.flatten()
    for img, ax, label in zip(images_arr, axes, lbls):
        print(label)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(str(label),fontsize=7)
    plt.tight_layout()
    plt.show


# In[7]:


import warnings
warnings.filterwarnings("ignore") 
plotImages(images,labels)


#     It seems like things work (hard to tell the difference of the big cats, but we see that the batch is shuffled and that labels are not all the same). The altered colors of the pictures are an artifact of the preprocessing we did before, in particular the same preprocessing of the images that the VGG16 model uses.

# ### 1. Implement a CNN, having 3 convolution layers. Determine the number and size of the filters and the rest of the hyperparameters.
# 
# ### 2. Define conveniently the model (optimization, loss, metric, ...).

# ### define function to create CNN model

# In[61]:


#we will create a sequential model with three layers and then apply a grid search to find the best hyperparams
#for this, we need to define a function that takes different ranges (for integer params) and discrete selections
#(choice of activation function for example) as input


#to increase performance (after knowing that everything runs smoothly):
#   - add one more layer
#   - try the sigmoid activation as well
#   - add dropout layer and different learning rates (try doing this when grid search is finished on current
#architecture in order to save time)

def creating_model(first_size,first_kernel_size,second_size,second_kernel_size,third_size,third_kernel_size,                   dropout_rate = 0.4):
    
    
    ##### NOT SURE WHY BUT THE GRIDSEARCH DOESN'T WANT TO TAKE A DICTIONARY AS INPUT, looks kind of ugly now ####
    
    #first_size         = params['first_size']
    #first_kernel_size  = params['first_kernel_size']
    
    #second_size        = params['second_size']
    #second_kernel_size = params['second_kernel_size']
    
    #third_size         = params['third_size']
    #third_kernel_size  = params['third_kernel_size']
    
    #act_func           = params['act_func']
    
    
    #initialize model
    model = Sequential()
    
    model.add(layers.Conv2D(
            filters = first_size,
            padding = 'same',
            kernel_size = (first_kernel_size,first_kernel_size),
            activation = 'relu',
            input_shape = (target_size[0],target_size[0],3)))
    
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(
            filters = second_size,
            padding = 'same',
            kernel_size = (second_kernel_size,second_kernel_size),
            activation = 'relu'))
        
    model.add(layers.Dropout(dropout_rate))    
    model.add(layers.MaxPooling2D(pool_size=(2,2)))


    model.add(layers.Flatten())
        
    #we have ten different classes to classify --> dense layer with 10 nodes using the softmax for activation
    model.add(layers.Dense(10, activation = 'softmax'))
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
              
    return model


# ### define hyperparameter space for the grid search

# In[62]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam

#increase the hyperparameter space eventually, want to make sure everything runs smoothly first

#activation_funcs = ['relu','sigmoid']
activation_funcs = ['relu']
#first_layer_size = [32,64,128]
first_layer_size = [32]
#first_layer_kernel_size = [3,5,7]
first_layer_kernel_size = [3]
#second_layer_size = [32,64,128,256]
second_layer_size = [64,128]
#second_layer_kernel_size = [3,5,7]
second_layer_kernel_size = [3]
third_layer_size = [64,128]
third_layer_kernel_size = [3]

params = {

    'first_size' : first_layer_size,
    'first_kernel_size' : first_layer_kernel_size,
    'second_size' : second_layer_size,
    'second_kernel_size' : second_layer_kernel_size,
    'third_size' : third_layer_size,
    'third_kernel_size' : third_layer_kernel_size
     
}


# ### preprocess data once more in order to decrease the computational expensiveness for the grid search process

# In[33]:


import itertools
train_subset_ratio = 0.5
valid_subset_ratio = 0.9

train_subset_size = int(len(train_batches) * train_subset_ratio)
valid_subset_size = int(len(valid_batches) * valid_subset_ratio)

train_batches_subset = itertools.islice(train_batches, train_subset_size)
valid_batches_subset = itertools.islice(valid_batches, valid_subset_size)

train_images_subset = []
train_labels_subset = []
for _ in range(train_subset_size):
    images, labels = next(train_batches_subset)
    train_images_subset.extend(images)
    train_labels_subset.extend(labels)

train_images_subset = np.array(train_images_subset)
train_labels_subset = np.array(train_labels_subset)

valid_images_subset = []
valid_labels_subset = []

for _ in range(valid_subset_size):
    images, labels = next(valid_batches_subset)
    valid_images_subset.extend(images)
    valid_labels_subset.extend(labels)

valid_images_subset = np.array(valid_images_subset)
valid_labels_subset = np.array(valid_labels_subset)


# ### carry out grid search

#     We set workers = -1 to parallellize the process and we set epochs to 3 only - hopefully this is enough to detect which set of parameters show signs of performing well and it saves a lof of computational cost. When training the model with these parameters at a later stage, we will increase the number of epochs to optimize performance. Update: This procedure did not work, instead, we removed the additional convolutional layer and bumped up number of epochs to 10 again.

# In[63]:


model = KerasClassifier(build_fn = creating_model, verbose = 2, epochs = 10)


# In[ ]:


#bumping into trouble when I try to use more CPUs, set n_jobs = -1 to use 4

grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 3, verbose = 2)

grid_search.fit(X = train_images_subset, y = train_labels_subset,                validation_data = (valid_images_subset,valid_labels_subset), workers = -1, verbose = 2)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# ## Best params

#     From the grid search (which took 50+ minutes, and this was done with a decreased hyperparameter space as well - was not able to run multiple CPUs for some reason), we obtain the optimal parameters:
# 
# ##### activation function: relu
# ##### kernel size first layer: 3 x 3
# ##### size first layer: 32
# ##### kernel size second layer: 3 x 3
# ##### size second layer 64
# 
# 
# 
#     Now, these parameters will be selected in order to retrain our model. This time, we will use a dropout layer to avoid overfitting and hopefully this will yield better generalization as the model is exposed to the validation set. Also, we will try to include another layer (to start with we only had two hidden layers since we were not not sure what was included in the definition of 3-layer CNN here, i.e. if we should include the output layer in the count as well). Since the training data is rather complex as we have 10 different classes of very similar animals, this might help.
# 
#     It should be noted that the hyperparameter space can be made infinitely large, so in order to balance computation time and performance, we have chosen to let certain params be, such as which optimizer to use (in this case we used Adam).

# In[9]:


from tensorflow.keras.optimizers import Adam


# In[21]:


#dropping 40% of the nodes from each neuron randomly in order to decrease proness to overfit
dropout_rate = 0.4


model = Sequential()


model.add(layers.Conv2D(
        filters = 32,
        padding = 'same',
        kernel_size = (3,3),
        activation = 'relu',
        input_shape = (target_size[0],target_size[0],3)))

model.add(layers.Dropout(dropout_rate)) 
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(
        filters = 64,
        padding = 'same',
        kernel_size = (3,3),
        activation = 'relu',
        name='last_conv_layer'))

model.add(layers.Dropout(dropout_rate)) 
model.add(layers.MaxPooling2D(pool_size=(2,2)))


model.add(layers.Flatten())

#we have ten different classes to classify --> dense layer with 10 nodes using the softmax for activation
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ### summary of the model

# In[22]:


model.summary()


# ### fit the new model, this time using all training data and validation data

# In[24]:


model.fit(train_batches, validation_data = valid_batches,         epochs = 5, verbose = 2, workers = -1)

#when exporting to HTML, set verbose = 0


#     Without adding an additional hidden layer, we ended up with 50% accuaracy after running the training for 5 epochs. We have not yet converged to optimaly classify the training data, suggesting that increasing the number of epochs can be beneficial. Nevertheless, considering the fact that we are classifying 10 classes, 50% is OK for now. If time allows, we should try adding another layer to the model.

#     Now that the model is trained, we save it (we will need the saved model later when applying the grad-CAM technique).

# In[25]:


#extension h5 is commonly used
model.save(train_path+str('.h5'))


# ### 3. Assess the performance of the CNN predicting the 10 wild big cat species of test images and obtain the confusion matrix.

# In[235]:


predictions = model.predict(x = test_batches, verbose = 2)


# In[274]:


#taking the most confident prediction for each test picture
predicted_lbls = np.argmax(predictions, axis = -1)


# In[237]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix


# ### confusion matrix
# 
#     need to find a way to display the values...

# In[298]:


class_labels = [label for label, index in test_batches.class_indices.items()]
cm = confusion_matrix(y_true = test_batches.classes, y_pred = predicted_lbls, labels = np.unique(test_batches.classes))
display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.unique(test_batches.classes))
display.plot()
plt.show()


# In[306]:


from sklearn.metrics import accuracy_score
true_lbls = test_batches.classes
accuracy = accuracy_score(true_lbls,predicted_lbls)
print('The accuracy score for the model is: ' + str(accuracy))


# ### 4. Use a pretrained convolutional network such as VGG16, VGG19, Inceptionv3, ResNet50, EfficientNet, ... for transfer learning to classify the 10 wild big cat species with fine tunnig.
# 
# ### 6. Assess the performance to predict the 10 categories of test images and obtain the confusion matrix. (Q.5 grad-CAM follow after Q.6)

#     We will choose the VGG16 model for this task. However, the VGG_16 model seems to only have been trained on:
#     
#     
#     - Ocelot
#     - Cheetah
#     - Jaguar
#     - Lions
#     - Snow leopard
#     - Tiger
#     
#     out of the 10 of our current classes. To adjust for this, we will create a new directory containing only these 6 classes. Since we will later freeze all the weights of the downloaded VGG16 and add a output layer of 10 nodes instead of 1000, which is the only layer that we will train and adjust the weights accordingly for, this measure will hopefully increase the performance of the model.
#    
#    

# In[429]:


#Download model
VGG_16 = tf.keras.applications.vgg16.VGG16()


# ### VGG_16 model summary

# In[430]:


VGG_16.summary()


#         Note that the output layer in the current model is built to predict up to 1000 different classes, which is not necessary in our case. Hence, we will make some adjustments to this architecture. First, we will convert this model into a sequential model (currently it's a model from Keras functional API). And thereafter, we will add an output layer of 10 nodes instead of 10000.

# In[431]:


model_VGG16 = Sequential()
for layer in VGG_16.layers[:-1]:
    model_VGG16.add(layer)


#     It is important that we freeze all the weight of the VGG_16 model, except for the last output layer that we just constructed. This will reduce training time extensively, and since the model probably has gone through the training for these classes, training the model again is not necessary. Therefore, we will make sure these waits cannot be updated when we train the model on our data. 

#     We will need to see if we can unfreeze to see if this boosts performance. At the moment the performance of the model is really bad. UPDATE: after playing around with the learning rate as well as taking a bigger chunk of the training data to train on (previously we used only half of the training data), the model now seems to perform very well. Therefore, we will use transfer learning as intended in the beginning and freeze all weights in the VGG16 model, add one output layer of 6 nodes, and train the model with the weights linked to this output layer trainable. It should also be noted that we made an effort to decrease the size of the matrices of the pictures, but since we are using a specific preprocessing function that the VGG16 model expects, this measure of decreasing computation time was abborted and we went on using 224x224 pixel size as before.

# In[433]:


for layer in model_VGG16.layers:
    layer.trainable = False


# In[434]:


model_VGG16.summary()


#     Now we can add the last layer with 6 (not 10 as before) nodes, which will be the only trainable layer in the model. We should keep in mind that this will by default increase our prediction accuracy, since the risk of predicting incorrect is automatically less prevalent.

# In[435]:


model_VGG16.add(layers.Dense(6, activation = 'softmax'))


# In[436]:


model_VGG16.summary()


#     Now that the adjustments are done, we will train the model as before. Note that we have 24,582 params that are trainable, which is a very small fraction of the total params.
#  

# In[437]:


#new paths for the updated data sets

train_path = '/Users/ianwallgren/Desktop/UPC/Courses/NeuralNetworks/Assignments/Assignment_2/BigCats10_VGG16/train'
valid_path = '/Users/ianwallgren/Desktop/UPC/Courses/NeuralNetworks/Assignments/Assignment_2/BigCats10_VGG16/valid'
test_path = '/Users/ianwallgren/Desktop/UPC/Courses/NeuralNetworks/Assignments/Assignment_2/BigCats10_VGG16/test'

#theese are the updated classes we have on disc, now only 6 classes instead of 10
classes = ['CHEETAH','JAGUAR','LIONS','OCELOT','SNOW LEOPARD','TIGER']


# In[438]:


#save only a fraction of the available training data and decrease the image size in order to speed up computation
small_target_size = (224,224)
#maybe need to increase this ratio to boost performance, let's see (update: we increased it and it helped a lot)
train_ratio = 0.95

train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input,                validation_split = 1-train_ratio).                flow_from_directory(directory = train_path, target_size = small_target_size, classes = classes,                batch_size = batch_size, subset='training')

valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).                flow_from_directory(directory = valid_path, target_size = small_target_size, classes = classes,                batch_size = batch_size)

#we want to set shuffle = False for the test batches in order to be able to generate a valid confusion matrix later
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).                flow_from_directory(directory = test_path, target_size = small_target_size, classes = classes,                batch_size = batch_size, shuffle = False)


# In[441]:


model_VGG16.compile(optimizer = Adam(learning_rate=0.0001),                     loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[442]:


#the performance is really bad atm. Make sure the last layer is trainable...
model_VGG16.fit(train_batches,validation_data = valid_batches,epochs = 5, verbose = 2, workers = -1)


# ### Predictions using our altered VGG_16 model

# In[445]:


predictions_vgg16 = model_VGG16.predict(x = test_batches, verbose = 2)
predicted_vgg16_lbls = np.argmax(predictions_vgg16, axis = -1)


# ### Confusion matrix

# In[446]:


class_labels = [label for label, index in test_batches.class_indices.items()]
cm = confusion_matrix(y_true = test_batches.classes, y_pred = predicted_vgg16_lbls, labels = np.unique(test_batches.classes))
display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.unique(test_batches.classes))
display.plot()
plt.show()


#     As expected, the confusion matrix shows no missclassifications.

# ### Accuracy

# In[447]:


true_lbls = test_batches.classes
accuracy = accuracy_score(true_lbls,predicted_vgg16_lbls)
print('The accuracy score for the model is: ' + str(accuracy))


#     We are able to predict with 100% accuracy! Early signs of this was visible when we trained the last layer of the model, as the performance of the validation set for each epoch (excluding the first epoch) was 100%. Since we only have 5 samples in the validation set, we have to take this with a grain of salt. 

# ### 5. Apply Grad-CAM technique in several differents species for explaining image.

# In[32]:


from tensorflow.keras.models import Model


#     We will choose species corresponding to classes 1, 2, and 3 for this task.
# 

#     First, we need to load the model that we created before.

# In[27]:


model_location = train_path+str('.h5')

model_loaded = keras.models.load_model(model_location)


#     Next, we create out grad-CAM model, which is our initial model up until the last convolutional layer.

# In[34]:


last_conv_layer = model_loaded.get_layer('last_conv_layer')
grad_model = Model(inputs=model_loaded.inputs, outputs=[last_conv_layer.output, model_loaded.output])


#     Now, we need to, for each of the feature maps in the last convolutional layer, calculate the gradient with respect to the loss function for all of the pixels. Then, we will reduce the spatial dimensions for all of these feature maps in the last layer to 1 by applying Gradient Average Pooling (GAP). This value (we will have one value for each of the feature maps in the last convolutional layer, will then be multiplied into the original feature map, highlighting the areas of the feature map the model found important in order to classify the object. The next step is to apply the ReLU activation function to all feature maps in order to get rid of any negative values. Then, we combine these altered feature maps, resize the dimensions to fit the original image, and then overlay this heatmap on the original image and plot the output.

# In[168]:


#picking out a random image to classify as well as its corresponding label
img,labels = next(train_batches)
img, labels = img[0], labels[0]


# In[169]:


plt.imshow(img)
display(labels)


#     This seems to be a puma according to the label. Puma has index 7, so we save this in a variable in order to be able to calculate the loss in the next step.

# In[170]:


index = 7


# In[171]:


#fixing the format before running the next step of the process
img_single = np.expand_dims(img, axis=0)


# In[172]:


img.shape


# In[184]:


#obtain the gradients
#the grad_model() returns as output both the convolutional output as well as the prediction

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_single)
    target_class_output = predictions[:, index]  # Assuming class index 0 is the target class
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(index, predictions)
grads = tape.gradient(target_class_output, conv_outputs)


# In[185]:


#now we apply the GAP (Global Average Pooling)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


# In[186]:


num_channels = last_conv_layer.output.shape[-1]
reshaped_pooled_grads = tf.reshape(pooled_grads, (1, 1, num_channels))


# In[187]:


#here we multiply each pixel in each of the feature maps 
#in the last convolutional layer with the importance weight calculated by the GAP procedure
heatmap = last_conv_layer.output * reshaped_pooled_grads
heatmap = tf.reduce_mean(heatmap, axis=-1, keepdims=True)


# In[194]:


heatmap


# In[195]:


import tensorflow.keras.backend as K
import tensorflow as tf

# Apply operations using TensorFlow operations
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap)
heatmap *= 255

# Convert the heatmap to a NumPy array (optional)
heatmap_array = K.eval(heatmap)
heatmap_array = K.eval(heatmap)


# ### 7. Compare both networks performance in 1) and 4). Use F1-measure among others performance measures.

#     After having trained both the networks in task 1) and 4), we can conclude that the pre-trained VGG16 model performs a lot better than our own model. However, one explanation for this could be the fact that we have used only two hidden layers in our current model, and considering the complexity of the data, this might be too little. Another reason for this is clearly the fact that the VGG16 model only predicts and trains on 6 different classes, while our model is exposed to predicting and training on all the classes. As mentioned before, the reason for this is that the VGG16 model is trianed on the ImageNet data set, and since this data set does not contain all 10 classes, it would not make sense to keep all 10 classes when training the last layer of the model. Maybe, the model would be able to adjust to this during training, which would actually be an interesting experiment to conduct. So if there is enough time, we might do such experiment as well.

#     The f1 performance measure:

# In[ ]:





# # Unfinished code

# In[ ]:


#We will try to use the Keras tuner in order to find the best selection of hyperparameters for our CNN model with
#three layers.

import keras_tuner
from tensorflow import keras
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers

model = KerasClassifier(build_fn = creating_model, verbose = 2, epochs = 10)

#first we need to create a function that returns a keras model
def build_model(hyperparams):
    model = keras.Sequential([
        
        keras.layers.Conv2D(
            filters = hyperparams.Int('first_filter', min_value = 32, max_value = 128, step = 2),
            kernel_size = hyperparams.Choice('first_kernel', values = [3,6]),
            activation = hyperparams.Choice('relu','sigmoid'),
            input_shape = (target_size[0],target_size[0],3)),
        
        keras.layers.Conv2D(
            filters = hyperparams.Int('second_filter', min_value = 64, max_value = 256, step = 2),
            kernel_size = hyperparams.Choice('second_kernel', values = [3,6]),
            activation = hyperparams.Choice('relu','sigmoid')),
        
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation = 'softmax')
        
    ])
    
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


#first we need to create a function that returns a keras model
def build_model(hyperparams):
    model = keras.Sequential()
        
    model.add(layers.Conv2D(
            filters = hyperparams.Int('first_filter', min_value = 32, max_value = 128, step = 2),
            kernel_size = hyperparams.Choice('first_kernel', values = [3,6]),
            activation = 'relu',
            input_shape = (target_size[0],target_size[0],3)))
        
    model.add(layers.Conv2D(
            filters = hyperparams.Int('second_filter', min_value = 64, max_value = 256, step = 2),
            kernel_size = hyperparams.Choice('second_kernel', values = [3,6]),
            activation = 'relu'))
        
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

#We will search for the optimal hyperparams by saving a part of our training set for validation (we have a seperate
#validation set on disc that we can use later, but considering the size of the training set, using a little bit
#of it for validation should not be a problem)
tuner = RandomSearch(build_model, objective = 'val_accuracy', max_trials = 5)

#now we use the tuner function and give it our test data as input to find the optimal hyperparameters
tuner.search(x = train_batches, validation_data = valid_batches, epochs = 5)


# In[ ]:


train_img = []
train_lbls = []
valid_img = []
valid_lbls = []



########################################## training data
for _ in range(len(train_batches)):
    images, labels = next(train_batches)
    train_img.extend(images)
    train_lbls.extend(labels)

#need to be of type array
train_img = np.array(train_img)
train_lbls = np.array(train_lbls)

########################################## validation data
for _ in range(len(valid_batches)):
    images, labels = next(valid_batches)
    valid_img.extend(images)
    valid_lbls.extend(labels)

#need to be of type array
valid_img = np.array(valid_img)
valid_lbls = np.array(valid_lbls)

assert len(train_img) == len(train_lbls)
assert len(valid_img) == len(valid_lbls)

