from itertools import count
import math
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plot
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
loaded_date = [];


def load_data():
    file_names = ["clean_fake.txt", "clean_real.txt"]

    vectorized = []
    length = []
    clean_fake_file = open(file_names[0]).readlines()
    clean_real_file = open(file_names[1]).readlines()
    vectorized.extend(clean_fake_file)
    length.append(len(clean_fake_file))
    vectorized.extend(clean_real_file)
    length.append(len(clean_real_file))
    vectorizer = CountVectorizer(analyzer='word')
    
    vectorized = [x.strip() for x in vectorized]
    clean_fake = vectorizer.fit_transform(vectorized)

    clean_real = np.array(np.ones(length[0]))
    temp = np.array(np.zeros(length[1]))

    clean_real = np.concatenate((clean_real, temp), axis=0)
    clean_real = clean_real.transpose()

    fake_train, fake_test, real_train, real_test = train_test_split(clean_fake, clean_real, test_size=0.15)
    fake_train, fake_validate, real_train, real_validate = train_test_split(clean_fake, clean_real, test_size=0.15)
    return [{"fake_train": fake_train, "fake_test": fake_test, "fake_val": fake_validate,
             "real_train": real_train, "real_test": real_test, "real_val": real_validate}, vectorizer,
            clean_fake, clean_real]


def select_tree_model(dataset):
    clafs = []
    max_depth = [10 * i for i in range(1, 11)]
    for i in max_depth:
        claf = DecisionTreeClassifier(max_depth=i, criterion="gini")
        clafs.append(claf)
        claf = DecisionTreeClassifier(max_depth=i, criterion="entropy")
        clafs.append(claf)
    accuracy_scores = []
    print("Depth\tCriteria\tAccuracy\n############################################")
    for claf in clafs:
        claf.fit(dataset[0]["fake_train"], dataset[0]["real_train"])
        real_pred = claf.predict(dataset[0]["fake_val"])
        accuracy_scores.append(metrics.accuracy_score(dataset[0]["real_val"], real_pred))
        print("%s\t\t%s\t\t%s" % (claf.max_depth, claf.criterion, accuracy_scores[-1]))
    max_accuracy_index = accuracy_scores.index(max(accuracy_scores))
    print(
        "\nDepth %s and %s criteria has the highest validation accuracy of %s!" % (
            clafs[max_accuracy_index].max_depth,
            clafs[max_accuracy_index].criterion,
            accuracy_scores[max_accuracy_index]))

    return clafs[max_accuracy_index]

def compute_information_gain(dataset):
    #I(Y|X) = H(Y) - H(Y|X)
    p_fake = np.count_nonzero(dataset[0]["real_train"] == 0) / len(
        dataset[0]["real_train"])
    p_real = np.count_nonzero(dataset[0]["real_train"] == 1) / len(
        dataset[0]["real_train"])

def select_knn_model(dataset):
    vali = []
    t_error = []
    kn = []
    for k in range(1, 21):
        neighbours = KNeighborsClassifier(n_neighbors=k)
        neighbours.fit(dataset[0]["fake_train"], dataset[0]["real_train"])
        kn.append(neighbours)
        val_predict = neighbours.predict(dataset[0]["fake_val"])
        vali.append(
            1 - sum(val_predict == dataset[0]["real_val"]) / len(
                dataset[0]["real_val"]))
        predict_train = neighbours.predict(dataset[0]["fake_train"])
        t_error.append(1 - sum(predict_train == dataset[0]["real_train"]) / len(
            dataset[0]["real_train"]))
    t_plot = []
    test_plot = []
    for i in t_error:
        t_plot.append(float(i))
    for i in vali:
        test_plot.append(float(i))
    df = pd.DataFrame({'train': t_plot, 'test': test_plot},
                      index=range(1, 21))
    df.plot()
    plot.show()
    return kn[vali.index(max(vali))]

if __name__ == '__main__':
    print("Hello world")
    loaded_data = load_data()
    ## best = select_tree_model(loaded_data)
    #compute_information_gain(loaded_data)
    #select_knn_model(loaded_data)

