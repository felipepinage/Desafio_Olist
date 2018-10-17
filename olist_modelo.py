import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib

# OVERSAMPLING
def oversampling(X_dados, y_labels):
    rus = SMOTE(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_dados, np.ravel(y_labels, order='C'))
    return [X_resampled, y_resampled]

# RESULTADOS
def resultados(labelsTeste, predicoes):
    acc = metrics.accuracy_score(labelsTeste, predicoes)
    fscore = metrics.f1_score(labelsTeste, predicoes, average='macro')
    prec = metrics.precision_score(labelsTeste, predicoes, average='macro')
    recall = metrics.recall_score(labelsTeste, predicoes, average='macro')
    file.write("\nEVALUATION METRICS\n"
               "acc: %0.4f - fscore: %0.4f - prec: %0.4f - recall: %0.4f\n" % (acc, fscore, prec, recall))
    return [acc, fscore, prec, recall]

# PLOTAR MATRIZ DE CONFUSÃO
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


df_prepared = pd.read_csv('olist_prepared.csv', sep=',', header=0).replace(np.NaN, 0)  #read csv file
file = open("olist_modelo_results.txt","w");

data = df_prepared.drop(['Unnamed: 0','most_voted_class'], axis=1) #separate data from labels
data = data.div(data.sum(axis=1), axis=0) #normalizate data to build the classifier

file.write("----- ORIGINAL MODEL -----\n")

features = list(data) #get the feature names
class_names = ['prob_entrega', 'prob_qualidade', 'satisfeito'] #set classes names
dataNoLabels = np.asmatrix(data) #data: dataframe to nArray
labels = np.asmatrix(df_prepared['most_voted_class']).transpose() #labels: dataframe to nArray
file.write("\nOriginal number of samples:\n"
            "%d - satisfeito com o pedido\n"
            "%d - problema de qualidade\n"
            "%d problema na entrega\n"
            % (np.count_nonzero(labels == 2), np.count_nonzero(labels == 1), np.count_nonzero(labels == 0)))

dataNoLabels, labels = oversampling(dataNoLabels, labels) #oversampling of minority labels
file.write("\nNumber of samples after oversampling:\n"
            "%d - satisfeito com o pedido\n"
            "%d - problema de qualidade\n"
            "%d problema na entrega\n"
            % (np.count_nonzero(labels == 2), np.count_nonzero(labels == 1), np.count_nonzero(labels == 0)))

X_train, X_test, y_train, y_test = train_test_split(dataNoLabels, labels, test_size=0.4, random_state=0) #split the dataset to train and test
clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_train, np.ravel(y_train,order='C')) #train RandomForest
predicoes = clf.predict(X_test) #test RandomForest
acc, f1s, precisao, recall = resultados(y_test, predicoes) #show results

c_mat = metrics.confusion_matrix(y_test, predicoes) #get the confusion matrix of original model
plt.figure()
plot_confusion_matrix(c_mat, classes=class_names, title='Confusion Matrix (Original Model)') #plot confusion matrix of original model

file.write("\n----- LIMITED MODEL (by features) -----\n")

file.write('\nFEATURE IMPORTANCES\n')
for feature in zip(list(data), clf.feature_importances_): #evaluate previous model to get feature importances
   file.write(str(feature) + "\n")

sfm = SelectFromModel(clf, threshold=0.038) #Select features with importance higher than 0.038
sfm.fit(X_train, np.ravel(y_train,order='C')) #train the selector

file.write('\nFEATURES SELECTED BY IMPORTANCE\n')
for feature_index in sfm.get_support(indices=True): #get most important features
    file.write(str(features[feature_index]) + "\n")

# Transform the data to create a new dataset containing only the most important features
X_important_train = sfm.transform(X_train) #new dataset to train
X_important_test = sfm.transform(X_test) #new dataset to test

clf_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1) #build a new classifier
clf_important.fit(X_important_train, np.ravel(y_train,order='C')) #train with new dataset

predicoes2 = clf_important.predict(X_important_test) #test with new dataset
acc2, f1s2, precisao2, recall2 = resultados(y_test, predicoes2) #show results

c_mat2 = metrics.confusion_matrix(y_test, predicoes2) #get the confusion matrix of limited model
plt.figure()
plot_confusion_matrix(c_mat2, classes=class_names, title='Confusion Matrix (Limited Model)') #plot confusion matrix of limited model

joblib.dump(clf_important, 'Olist_Model.pkl') #save the new model (limited) as pkl file

file.close()
