from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve, roc_curve
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

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


def xgboostForCVwithLossError(np_train, np_val, n_estimators, learning_rate, n_jobs):
    """全部默认参数下，根据loss和error确定learning_rate和n_estimators"""
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")
    eval_set = [(x_train, y_train), (x_val, y_val)]

    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, eval_metric=["error", "logloss"], random_state=10)
    clf.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    results = clf.evals_result()
    train_error = results["validation_0"]["error"]
    val_error = results["validation_1"]["error"]
    train_loss = results["validation_0"]["logloss"]
    val_loss = results["validation_1"]["logloss"]

    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return train_loss, val_loss, train_error, val_error, "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_


def xgboostForCVotherparameters(np_train, np_val, n_estimators, learning_rate, n_jobs, max_depth, min_child_weight, gamma, subsample, colsample_bytree):
    """确定了learning_rate和n_estimators之后，根据AUROC和AUPRC确定max_depth、min_child_weight、gamma、subsample和colsample_bytree"""
    x_train, y_train = np_train[:,:-1], np_train[:,-1].astype("int")
    x_val, y_val = np_val[:,:-1], np_val[:,-1].astype("int")

    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree, n_jobs=n_jobs, random_state=10)
    clf.fit(x_train, y_train)
    y_pred_label = clf.predict(x_val)
    y_pred_score = np.array(clf.predict_proba(x_val))[:, 1]
    accuracy, precision, recall, f1_score, auroc, auprc, *_ = metrics(y_true=y_val, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "xgboost", accuracy, precision, recall, f1_score, auroc, auprc



def xgboostForMultiTest(np_train, np_test, n_jobs, pred_score_out=False, specificity=None):
    x_train, y_train, x_test, y_test = np_train[:, :-1], np_train[:, -1].astype("int"), np_test[:, :-1], np_test[:, -1].astype("int")
    clf = XGBClassifier(n_estimators=600, learning_rate=0.03, n_jobs=n_jobs, max_depth=7, min_child_weight=0.1, gamma=0, subsample=0.9, colsample_bytree=1.0, random_state=10)
    clf.fit(x_train, y_train)
    y_pred_score = np.array(clf.predict_proba(x_test))[:, 1]
    if specificity is not None:
        if specificity==0.90:
            y_pred_label = np.array([1.0 if v>0.999638 else 0.0 for v in y_pred_score])
        elif specificity==0.95:
            y_pred_label = np.array([1.0 if v>0.999776 else 0.0 for v in y_pred_score])
        elif specificity==0.99:
            y_pred_label = np.array([1.0 if v>0.999856 else 0.0 for v in y_pred_score])
    else:
        y_pred_label = clf.predict(x_test)
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_test, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    if pred_score_out:
        return y_pred_score, "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_
    else:
        return "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_



def xgboostForBanchmarkDataset(np_train, np_test, n_jobs, specificity=None):
    x_train, y_train, x_test, y_test = np_train[:, :-1], np_train[:, -1].astype("int"), np_test[:, :-1], np_test[:, -1].astype("int")
    clf = XGBClassifier(n_estimators=600, learning_rate=0.03, n_jobs=n_jobs, max_depth=7, min_child_weight=0.1, gamma=0, subsample=0.9, colsample_bytree=1.0, random_state=10)
    clf.fit(x_train, y_train)
    y_pred_score = np.array(clf.predict_proba(x_test))[:, 1]
    if specificity is not None:
        if specificity==0.90:
            y_pred_label = np.array([1.0 if v>0.999638 else 0.0 for v in y_pred_score])
        elif specificity==0.95:
            y_pred_label = np.array([1.0 if v>0.999776 else 0.0 for v in y_pred_score])
        elif specificity==0.99:
            y_pred_label = np.array([1.0 if v>0.999856 else 0.0 for v in y_pred_score])
    else:
        y_pred_label = clf.predict(x_test)
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_test, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    
    return "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_



def xgboostForAblationExperiment(np_train, np_test, n_jobs):
    x_train, y_train, x_test, y_test = np_train[:, :-1], np_train[:, -1].astype("int"), np_test[:, :-1], np_test[:, -1].astype("int")
    clf = XGBClassifier(n_estimators=600, learning_rate=0.03, n_jobs=n_jobs, max_depth=7, min_child_weight=0.1, gamma=0, subsample=0.9, colsample_bytree=1.0, random_state=10)
    clf.fit(x_train, y_train)
    y_pred_score = np.array(clf.predict_proba(x_test))[:, 1]
    y_pred_label = clf.predict(x_test)
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_test, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    return "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_



def xgboostForRbpReceptorPrediction(np_train, np_test, n_jobs, specificity=None, predictedscore=False):
    x_train, y_train, x_test, y_test = np_train[:, :-1], np_train[:, -1].astype("int"), np_test[:, :-1], np_test[:, -1].astype("int")
    clf = XGBClassifier(n_estimators=600, learning_rate=0.03, n_jobs=n_jobs, max_depth=7, min_child_weight=0.1, gamma=0, subsample=0.9, colsample_bytree=1.0, random_state=10)
    clf.fit(x_train, y_train)
    y_pred_score = np.array(clf.predict_proba(x_test))[:, 1]
    if specificity is not None:
        if specificity==0.90:
            y_pred_label = np.array([1.0 if v>0.999638 else 0.0 for v in y_pred_score])
        elif specificity==0.95:
            y_pred_label = np.array([1.0 if v>0.999776 else 0.0 for v in y_pred_score])
        elif specificity==0.99:
            y_pred_label = np.array([1.0 if v>0.999856 else 0.0 for v in y_pred_score])
    else:
        y_pred_label = clf.predict(x_test)
    accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_ = metrics(y_true=y_test, y_pred_label=y_pred_label, y_pred_score=y_pred_score)
    if predictedscore:
        y_pred_score_list = y_pred_score.tolist()
        return y_pred_score_list
    else:
        return "xgboost", accuracy, precision, recall, f1_score, auroc, auprc, fpr, tpr, precision_, recall_


