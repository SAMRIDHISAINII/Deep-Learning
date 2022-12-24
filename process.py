# Applying different feature extraction techniques

import warnings
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings any
warnings.filterwarnings('ignore')

# To decide display window width on console
DESIRED_WIDTH = 320
pd.set_option('display.width', DESIRED_WIDTH)
np.set_printoptions(linewidth=DESIRED_WIDTH)
pd.set_option('display.max_columns', 30)

INPUTTRAINFILE = "data/segmentation.test"
INPUTTESTFILE = "data/segmentation.data"
TRAINFILE = "data/train.csv"
TESTFILE = "data/test.csv"

ATTRIBUTES = None


def filldatasetfile(inputfile, outputfile):
    """
        Creates the CSV File
    :param inputfile:
    :param outputfile:
    :return:
    """
    global ATTRIBUTES
    nfirstlines = []
    with open(inputfile) as _inp, open(outputfile, "w") as out:
        for i in range(5):
            if i == 3:
                ATTRIBUTES = ['LABELS'] + next(_inp).rstrip('\n').split(',')
            else:
                nfirstlines.append(next(_inp))
        for line in _inp:
            out.write(line)


def extractdata():
    """
        Extract and return the segmentation data in pandas dataframe format
    :return:
    """
    np.random.seed(0)
    if os.path.exists(TRAINFILE):
        os.remove(TRAINFILE)
    if os.path.exists(TESTFILE):
        os.remove(TESTFILE)

    filldatasetfile(INPUTTRAINFILE, TRAINFILE)
    filldatasetfile(INPUTTESTFILE, TESTFILE)

    # Convert csv to pandas dataframe
    traindata = pd.read_csv("data/train.csv", header=None)
    testdata = pd.read_csv("data/test.csv", header=None)
    traindata.columns = testdata.columns = ATTRIBUTES

    # Shuffle the dataframe
    traindata = traindata.sample(frac=1).reset_index(drop=True)
    testdata = testdata.sample(frac=1).reset_index(drop=True)

    return traindata, testdata


def preprocessdata(data):
    """
        Preprocess the data with StandardScalar and Label Encoder
    :param data: input dataframe of training or test set
    """
    labels = data['LABELS']
    features = data.drop(['LABELS'], axis=1)
    columns = features.columns
    enc = LabelEncoder()
    enc.fit(labels)
    labels = enc.transform(labels)
    features = StandardScaler().fit_transform(features)
    return features, labels, columns, data['LABELS']


def applyrandomforest(trainX, testX, trainY, testY):
    """
        Apply Random forest on input dataset.
    """
    start = time.process_time()
    forest = RandomForestClassifier(n_estimators=700, max_features='sqrt', max_depth=15)
    forest.fit(trainX, trainY)
    print("Time Elapsed: %s secs" % (time.process_time() - start))
    prediction = forest.predict(testX)
    print("Classification Report after applying Random Forest: ")
    print("----------------------------------------------------")
    print(classification_report(testY, prediction))


def applypca(trainX, testX, trainY, testY, columns):
    """
        Apply PCA on the dataset
    """
    # Fitting the PCA algorithm with our Data
    pca = PCA()
    pca.fit(trainX)
    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')  # for each component
    plt.title('Segmentation Dataset Explained Variance')
    plt.show(block=True)

    pca = PCA(n_components=12)
    pca.fit(Xtrain)
    trainX_pca = pca.transform(trainX)
    testX_pca = pca.transform(testX)
    applyrandomforest(trainX_pca, testX_pca, trainY, testY)

    # Visualizing the PCA coefficients using a heat map
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks(range(12), range(12))
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=60, ha='left')
    plt.xlabel('Feature')
    plt.ylabel('Principal Components')
    plt.show(block=True)


def applylda(trainX, testX, trainY, testY, actual_labels):
    lda = LinearDiscriminantAnalysis()
    lda.fit(trainX, trainY)
    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(lda.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')  # for each component
    plt.title('Segmentation Dataset Explained Variance')
    plt.show(block=True)
    lda = LinearDiscriminantAnalysis(n_components=5)
    lda.fit(trainX, trainY)
    trainX_lda = lda.transform(trainX)
    testX_lda = lda.transform(testX)
    # Plot Pairwise relationship between LDA components
    plt.figure(figsize=(10, 8), dpi=80)
    visualizedf = pd.DataFrame(trainX_lda, columns=['LDA1', 'LDA2', 'LDA3', 'LDA4', 'LDA5'])
    visualizedf = pd.concat([visualizedf, pd.DataFrame(actual_labels, columns=['LABELS'])], axis=1)
    print(visualizedf.sample(n=5))
    sns.pairplot(visualizedf, vars=visualizedf.columns[:-1], hue="LABELS", palette="husl")
    plt.show(block=True)
    applyrandomforest(trainX_lda, testX_lda, trainY, testY)


def applytsne(trainX, trainY):
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(trainX, trainY)
    # transform the data onto the first two principal components
    trainX_lda = lda.transform(trainX)
    colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
              "#A83683", "#4E655E"]
    plt.figure(figsize=(10, 10))
    plt.xlim(trainX_lda[:, 0].min(), trainX_lda[:, 0].max())
    plt.ylim(trainX_lda[:, 1].min(), trainX_lda[:, 1].max())
    for i in range(len(trainX_lda)):
        # actually plot the digits as text instead of using scatter
        plt.text(trainX_lda[i, 0], trainX_lda[i, 1], str(trainY[i]),
                 color=colors[trainY[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("LDA 0")
    plt.ylabel("LDA 1")
    plt.show(block=True)

    # Apply tSNE from Manifold learning for better visualization
    tsne = TSNE(random_state=21)
    # use fit_transform instead of fit, as TSNE has no transform method
    trainX_tsne = tsne.fit_transform(trainX)
    plt.figure(figsize=(10, 10))
    plt.xlim(trainX_tsne[:, 0].min(), trainX_tsne[:, 0].max())
    plt.ylim(trainX_tsne[:, 1].min(), trainX_tsne[:, 1].max())
    for i in range(len(trainX_tsne)):
        # actually plot the digits as text instead of using scatter
        plt.text(trainX_tsne[i, 0], trainX_tsne[i, 1], str(trainY[i]),
                 color=colors[trainY[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.show(block=True)


TRAINDATA, TESTDATA = extractdata()
print(TRAINDATA.head(n=5))
Xtrain, Ytrain, COLUMNS, ACTUAL_LABELS = preprocessdata(TRAINDATA)
Xtest, Ytest, _, _ = preprocessdata(TESTDATA)
applyrandomforest(Xtrain, Xtest, Ytrain, Ytest)
applypca(Xtrain, Xtest, Ytrain, Ytest, columns=COLUMNS)
applylda(Xtrain, Xtest, Ytrain, Ytest, actual_labels=ACTUAL_LABELS)
applytsne(Xtrain, Ytrain)
