from matplotlib import pyplot as plt
import numpy as np
from keras.models import model_from_json
import pandas as pd
import os
from os import system
from keras.preprocessing import image


########
# VIZU #
########
def lastErrors(XTRAIN,model,size=1000):
    Xviz = XTRAIN[:size]
    lab = train_labels[:size]
    p = model.predict(Xviz)
    pr = [cut_half(x) for x in p]
    i=0
    X = []
    l = []
    for reel, devine in zip(lab,pr):
        if reel != devine:
            X.append(Xviz[i])
            l.append(reel)
        i+=1
    return np.array(X), l

def visualize_incorrect_labels(x_data, y_real):
    count = 0
    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))
    figure = plt.figure(figsize=((maximum_square * 2,maximum_square * 2)))
    for i in range(x_data.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        plt.imshow(x_data[i, :, :, :])
        plt.axis('off')
        plt.title(" Real: " + str(int(y_real[i])), fontsize=10)
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



################
# LOADING DATA #
################
def loadDATA(PATH,full=False, some=False, little=10):
    #PATH = os.getcwd()
    data_train_path_target = "data_airbus_defi/train/"
    data_test_path = "data_airbus_defi/test/"

    ###############
    # SIZE LOADED #
    ###############
    if full:
        training_size = float("inf")
        testing_size = float("inf")
    elif some:
        training_size = 5000
        testing_size = float("inf")
    else:
        training_size = little
        testing_size = little



    #########
    # TRAIN #
    #########

    train_path = PATH +"/" + data_train_path_target + "target/"
    train_data_target = os.listdir(train_path)
    x_train = []
    tdt = 0
    for sample in (train_data_target):
        img_path = train_path+sample
        x = image.load_img(img_path)
        x_train.append(np.array(x))
        tdt += 1
        if tdt > training_size:
            break

    train_path = PATH +"/" + data_train_path_target + "other/"
    train_data_other = os.listdir(train_path)
    x_train2 = []
    tdo = 0
    for sample in (train_data_other):
        img_path = train_path+sample
        x = image.load_img(img_path)
        # preprocessing if required
        x_train2.append(np.array(x))
        tdo += 1
        if tdo > training_size:
            break


    ########
    # TEST #
    ########

    test_path = PATH+'/data_airbus_defi/test/'
    test_data = os.listdir(test_path)
    x_test = []
    td = 0
    output = pd.DataFrame(columns=["name"])
    test_data = [str(x) + ".jpg" for x in range(len(test_data))]
    for sample in (test_data):
        output.append({"name": sample},ignore_index=True)
        #print(sample)
        img_path = test_path+sample
        x = image.load_img(img_path)
        # preprocessing if required
        x_test.append(np.array(x))
        td+=1
        if td > testing_size:
            break


    # finally converting list into numpy array
    x_train = np.array(x_train)
    x_train2 = np.array(x_train2)
    x_test = np.array(x_test) / 255.
    XTRAIN = np.concatenate((x_train, x_train2), axis=0) / 255.
    train_label1 = np.array([1] * x_train.shape[0] + [0] * x_train2.shape[0])
    
    return XTRAIN, train_label1, x_test


def soumissionCSV(prediction, name,PATH):
    print("e")
    test_path = PATH+'/data_airbus_defi/test/'
    test_data = os.listdir(test_path)
    X = pd.DataFrame()
    X["name"] = test_data
    X["prediction"] = prediction
    X.set_index("name", inplace=True)
    X.to_csv("soumissions/"+name + ".csv", sep=";")
    print("CSV file written")

def prediction_from_model(model,x_test):
    pred = model.predict(x_test)
    prediction = [cut_half(x) for x in pred]
    print("some predictions")
    print(prediction[:9])
    return prediction, pred