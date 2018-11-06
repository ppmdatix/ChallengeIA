from matplotlib import pyplot as plt
import numpy as np
from keras.models import model_from_json


########
# VIZU #
########
def visualize_incorrect_labels(x_data, y_real, y_predicted):
    count = 0
    figure = plt.figure()
    incorrect_label_indices = (y_real != y_predicted)
    y_real = y_real[incorrect_label_indices]
    y_predicted = y_predicted[incorrect_label_indices]
    x_data = x_data[incorrect_label_indices, :, :, :]

    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

    for i in range(x_data.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        plt.imshow(x_data[i, :, :, :])
        plt.axis('off')
        plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize=10)

    plt.show()
    
def plotLearning(history):
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.close()

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.close()
    
    plt.plot(history.history['val_acc'])
    plt.title('Validation Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('Epoch')
    plt.show()
    plt.close()
    
###############
# KERAS MODEL #
###############

def saveModel(model, name):
    path = "model/" + name 
    model_json = model.to_json()
    with open(path + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path + ".h5")
    print("Saved model")

def loadModel( name):
    path = "model/" + name 
    json_file = open(path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + ".h5")
    #loaded_model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model
    
    
######################
# DATA PREPROCESSING #
######################
def cut_half(x):
    if x < .5:
        return 0
    else:
        return 1

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



##############
# PREDICTION #
##############


