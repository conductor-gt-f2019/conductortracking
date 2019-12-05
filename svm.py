from sklearn.neighbors import KNeighborsClassifier
import glob
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from matplotlib import pyplot as plt

def getData(save = False):
    train = 'train'
    val = 'val'
    data_f = 'data'
    data_folders = ['train', 'val']

    X = {}
    y = {}
    for data_folder in data_folders:
        df = os.path.join(data_f, data_folder)
        tt = os.listdir(df)
        if ".DS_Store" in tt:
            tt.remove(".DS_Store")

        files = {}

        classes = os.listdir(df)
        len_data = 0
        for c in classes:
            files[c] = glob.glob(os.path.join(df, c, '*.png'))
            len_data += len(files[c])

        scale = .25
        data = np.zeros((len_data, int(500*scale), int(500*scale)))
        labels = np.zeros(len_data)
        class_id = {'2_2': 0, '3_4': 1, '4_4': 2}

        index = 0
        for c in classes:
            for f in files[c]:
                labels[index] = class_id[c]
                data[index] = rescale(rgb2gray(imread(f)), scale)
                # plt.imshow(data[index])
                # plt.show()
                index += 1

        train_data = data.reshape(len_data, -1)
        print(data_folder, train_data.shape, labels.shape)
        X[data_folder] = train_data
        y[data_folder] = labels

    X_train = X['train']
    y_train = y['train']
    X_val = X['val']
    y_val = y['val']

    if save:
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)
    
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = getData(True)

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

print("building classifier...")
knn = SVC(gamma='auto')
knn.fit(X_train, y_train)

print("predicting...")
preds = knn.predict(X_val)
accuracy = np.sum(preds == y_val)/len(y_val)
print("accuracy:", accuracy)
print(confusion_matrix(y_val, preds))