from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve, roc_curve
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def metrics(y_true, y_pred_label, y_pred_score):
    """
        y_pred: predicted label of y
    """
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_label)
    precision = precision_score(y_true=y_true, y_pred=y_pred_label)
    recall = recall_score(y_true=y_true, y_pred=y_pred_label)
    f1_s = f1_score(y_true=y_true, y_pred=y_pred_label)
    
    ## auroc & auprc
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred_score)
    auroc = auc(fpr, tpr)
    precision_, recall_, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred_score, pos_label=1)
    auprc = auc(recall_, precision_)

    return accuracy, precision, recall, f1_s, auroc, auprc,\
           list(fpr), list(tpr), list(precision_), list(recall_)


def basemodelnaivebayes(np_train, np_val):
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "naivebayes", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_


def basemodelknn(np_train, np_val, n_neighbors, n_jobs):
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "knn", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_


def basemodelrandomforest(np_train, np_val, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, n_jobs):
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs, random_state=20)
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "randomforest", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_



def basemodeladaboost(np_train, np_val, n_estimators, learning_rate):
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=20)
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "adaboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_


def basemodelsvm(np_train, np_val, C, gamma):
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    clf = SVC(C=C, gamma=gamma, random_state=20, probability=True)
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "svm", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_


def basemodelxgboost(np_train, np_val, n_estimators, learning_rate, n_jobs):
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs, random_state=10)
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_

