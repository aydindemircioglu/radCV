#!/usr/bin/python3

from datetime import datetime
import pandas as pd
import random
import numpy as np
import time
#import pycm
import shutil
import pathlib
import os
import math
import sys
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time
import copy
import random
import pickle
from joblib import Parallel, delayed
import tempfile
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
import itertools
import multiprocessing
import socket
from glob import glob
from collections import OrderedDict
import logging
import mlflow
from typing import Dict, Any
import hashlib
import json


from pymrmre import mrmr
from pprint import pprint
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
from skfeature.function.statistical_based import f_score, t_score
from skfeature.function.similarity_based import reliefF
from sklearn.feature_selection import mutual_info_classif

from mlflow import log_metric, log_param, log_artifact, log_dict

from loadData import *
from utils import *



### parameters
TrackingPath = "/data/radCV/mlruns"


# NOTE: if N get larger than 64, mrmre must be adapted to it.
# we need to fix the #solutions there, because else it has sup-qaudratic runtime
# and needs days for each round if used with 7000 features
fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32],
        "Methods": {
            "LASSO": {"C": np.logspace(-1, 3, 5, base = 10.0)},
            "SVMRFE": {},
            "ReliefF": {},
            "MIM": {},
            "MRMRe": {},
            "tScore": {},
            "fScore": {}
        }
    }
})

clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "Constant": {},
            "RBFSVM": {"C":np.logspace(-6, 6, 13, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [25, 100, 250]},
            "XGBoost": {"learning_rate": [0.001, 0.1, 0.3, 0.9], "n_estimators": [25, 100, 250]},
            "LogisticRegression": {},
            "NeuralNetwork": {"layer_1": [8, 16, 32], "layer_2": [8, 16, 32]},
            "NaiveBayes": {}
        }
    }
})



#    wie CV: alle parameter gehen einmal durch
def getExperiments (experimentList, expParameters, sKey, inject = None):
    newList = []
    for exp in experimentList:
        for cmb in list(itertools.product(*expParameters.values())):
            pcmb = dict(zip(expParameters.keys(), cmb))
            if inject is not None:
                pcmb.update(inject)
            _exp = exp.copy()
            _exp.append((sKey, pcmb))
            newList.append(_exp)
    experimentList = newList.copy()
    return experimentList



# this is pretty non-generic, maybe there is a better way, for now it works.
def generateAllExperiments (experimentParameters, verbose = False):
    experimentList = [ [] ]
    for k in experimentParameters.keys():
        if verbose == True:
            print ("Adding", k)
        if k == "BlockingStrategy":
            newList = []
            blk = experimentParameters[k].copy()
            newList.extend(getExperiments (experimentList, blk, k))
            experimentList = newList.copy()
        elif k == "FeatureSelection":
            # this is for each N too
            print ("Adding feature selection")
            newList = []
            for n in experimentParameters[k]["N"]:
                for m in experimentParameters[k]["Methods"]:
                    fmethod = experimentParameters[k]["Methods"][m].copy()
                    fmethod["nFeatures"] = [n]
                    newList.extend(getExperiments (experimentList, fmethod, m))
            experimentList = newList.copy()
        elif k == "Classification":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        else:
            experimentList = getExperiments (experimentList, experimentParameters[k], k)

    return experimentList



# if we do not want scaling to be performed on all data,
# we need to save thet scaler. same for imputer.
def preprocessData (X, y):
    simp = SimpleImputer(strategy="mean")
    X = pd.DataFrame(simp.fit_transform(X),columns = X.columns)

    sscal = StandardScaler()
    X = pd.DataFrame(sscal.fit_transform(X),columns = X.columns)
    return X, y



def applyFS (X, y, fExp):
    print ("Applying", fExp)
    return X, y



def applyCLF (X, y, cExp, fExp = None):
    print ("Training", cExp, "on FS:", fExp)
    return "model"



def testModel (y_pred, y_true, idx, fold = None):
    t = np.array(y_true)
    p = np.array(y_pred)

    # naive bayes can produce nan-- on ramella2018 it happens.
    # in that case we replace nans by 0
    p = np.nan_to_num(p)
    y_pred_int = [int(k>=0.5) for k in p]

    acc = accuracy_score(t, y_pred_int)
    df = pd.DataFrame ({"y_true": t, "y_pred": p}, index = idx)

    return {"y_pred": p, "y_test": t,
                "y_pred_int": y_pred_int,
                "idx": np.array(idx).tolist()}, df, acc



def getRunID (pDict):
    def dict_hash(dictionary: Dict[str, Any]) -> str:
        dhash = hashlib.md5()
        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    run_id = dict_hash(pDict)
    return run_id



def getAUCCurve (modelStats, dpi = 100):
    # compute roc and auc
    fpr, tpr, thresholds = roc_curve (modelStats["y_test"], modelStats["y_pred"])
    area_under_curve = auc (fpr, tpr)
    if (math.isnan(area_under_curve) == True):
        print ("ERROR: Unable to compute AUC of ROC curve. NaN detected!")
        print (modelStats["y_test"])
        print (modelStats["y_pred"])
        raise Exception ("Unable to compute AUC")
    sens, spec = findOptimalCutoff (fpr, tpr, thresholds)


    f, ax = plt.subplots(figsize = (6,6), dpi = dpi)
    ax.plot(fpr, tpr, 'b')
    #plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_aspect('equal', 'datalim')

    return (f, ax), area_under_curve, sens, spec



def getPRCurve (modelStats, dpi = 100):
    # compute roc and auc
    precision, recall, thresholds = precision_recall_curve(modelStats["y_test"], modelStats["y_pred"])
    try:
        f1 = f1_score (modelStats["y_test"], modelStats["y_pred_int"])
    except Exception as e:
        print (modelStats["y_test"])
        print (modelStats["y_pred_int"])
        raise (e)
    f1_auc = auc (recall, precision)
    if (math.isnan(f1_auc) == True):
        print ("ERROR: Unable to compute AUC of PR curve. NaN detected!")
        print (modelStats["y_test"])
        print (modelStats["y_pred"])
        raise Exception ("Unable to compute AUC")


    f, ax = plt.subplots(figsize = (6,6), dpi = dpi)
    ax.plot(recall, precision, 'b')
    #plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_aspect('equal', 'datalim')

    return (f, ax), f1, f1_auc



def logMetrics (foldStats):
    #print ("Computing stats over all folds", len([f for f in foldStats if "fold" in f]))
    y_preds = []
    y_test = []
    y_index = []
    #pprint (foldStats)
    for k in foldStats:
        if "fold" in k:
            y_preds.extend(foldStats[k]["y_pred"])
            y_test.extend(foldStats[k]["y_test"])
            y_index.extend(foldStats[k]["idx"])

    modelStats, df, acc = testModel (y_preds, y_test, idx = y_index, fold = "ALL")
    (f_ROC, ax_ROC), roc_auc, sens, spec = getAUCCurve (modelStats, dpi = 72)
    (f_PR, ax_PR), f1, f1_auc = getPRCurve (modelStats, dpi = 72)

    expVersion = '_'.join([k for k in foldStats["params"] if "Experiment" not in k])
    pID = str(foldStats["params"])

    # register run in mlflow now
    run_id = getRunID (foldStats["params"])
    with mlflow.start_run(run_name = run_id, tags = {"Version": expVersion, "pID": pID}) as run:
        # this is stupid, but well, log a file with name=runid
        log_dict(foldStats["params"], run_id+".ID")
        log_dict(modelStats, "params.yml")
        log_metric ("Accuracy", acc)
        log_metric ("Sens", sens)
        log_metric ("Spec", spec)
        log_metric ("AUC", roc_auc)
        log_metric ("F1", f1)
        log_metric ("F1_AUC", f1_auc)
        #print (foldStats["features"])
        log_dict(foldStats["features"], "features.json")
        for k in foldStats["params"]:
            log_param (k, foldStats["params"][k])
        with tempfile.TemporaryDirectory() as temp_dir:
            predFile = os.path.join(temp_dir, "preds.csv")
            df.to_csv(predFile)
            mlflow.log_artifact(predFile)

            rocFile = os.path.join(temp_dir, "ROC.png")
            f_ROC.savefig(rocFile)
            mlflow.log_artifact(rocFile)
            prFile = os.path.join(temp_dir, "PR.png")
            f_PR.savefig(prFile)
            mlflow.log_artifact(prFile)
    plt.close(fig=f_PR)
    plt.close(fig=f_ROC)
    print(".", end = '', flush=True)
    return {}



def createFSel (fExp):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter=500, solver='liblinear', C = C)
        #print ("LASSO with C=",C)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures)


    if method == "ReliefF":
        def relieff_score_fct (X, y):
            scores = reliefF.reliefF (X,y)
            return scores
        pipe = SelectKBest(relieff_score_fct, k = nFeatures)


    if method == "MIM":
        pipe = SelectKBest(mutual_info_classif, k = nFeatures)


    if method == "MRMRe":
        def mrmr_score_fct (X, y):
            Xp = pd.DataFrame(X, columns = range(X.shape[1]))
            yp = pd.DataFrame(y, columns=['Target'])

            solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=64+6, solution_count=5)
            scores = [0]*Xp.shape[1]
            for k in solutions.iloc[0]:
                for j, z in enumerate(k):
                    scores[z] = scores[z] + Xp.shape[1] - j
            scores = np.asarray(scores, dtype = np.float32)
            scores = scores/np.sum(scores)
            return scores
        pipe = SelectKBest(mrmr_score_fct, k = nFeatures)


    if method == "SVMRFE":
        def svmrfe_score_fct (X, y):
            svc = LinearSVC (C=1)
            rfe = RFECV(estimator=svc, step=0.10, scoring='roc_auc', n_jobs=1)
            rfe.fit(X, y)
            scores = rfe.ranking_
            return scores
        pipe = SelectKBest(svmrfe_score_fct, k = nFeatures)


    if method == "fScore":
        def f_score_fct (X, y):
            scores = f_score.f_score (X,y)
            return scores
        pipe = SelectKBest(f_score_fct, k = nFeatures)


    if method == "tScore":
        def t_score_fct (X, y):
            scores = t_score.t_score (X,y)
            return scores
        pipe = SelectKBest(t_score_fct, k = nFeatures)

    return pipe



def createClf (cExp):
    #print (cExp)
    method = cExp[0][0]
    if method == "Constant":
        model = DummyClassifier()

    if method == "RBFSVM":
        C = cExp[0][1]["C"]
        g = cExp[0][1]["gamma"]
        model = SVC(kernel = "rbf", C = C, gamma = g, probability = True)

    if method == "LogisticRegression":
        model = LogisticRegression(solver = 'lbfgs', random_state = 42)

    if method == "LinearSVM":
        alpha = cExp[0][1]["alpha"]
        model = SGDClassifier(alpha = alpha, loss = "log")

    if method == "RandomForest":
        n_estimators = cExp[0][1]["n_estimators"]
        model = RandomForestClassifier(n_estimators = n_estimators)

    if method == "XGBoost":
        learning_rate = cExp[0][1]["learning_rate"]
        n_estimators = cExp[0][1]["n_estimators"]
        model = XGBClassifier(learning_rate = learning_rate, n_estimators = n_estimators, use_label_encoder=False, eval_metric = "logloss", tree_method='gpu_hist', random_state = 42)

    if method == "NaiveBayes":
        model = GaussianNB()

    if method == "NeuralNetwork":
        N1 = cExp[0][1]["layer_1"]
        N2 = cExp[0][1]["layer_2"]
        #alpha = cExp[0][1]["alpha"]
        model = MLPClassifier (hidden_layer_sizes=(N1,N2,), random_state=42, max_iter = 1000)
    return model



@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
def executeExperiment (fselExperiments, clfExperiments, data, dataID):
    mlflow.set_tracking_uri(TrackingPath)

    y = data["Target"]
    X = data.drop(["Target"], axis = 1)
    X, y = preprocessData (X, y)

    # need a fixed set of folds to be comparable
    kfolds = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 1, random_state = 42)


    # experiment A
    raceOK = False
    while raceOK == False:
        try:
            mlflow.set_experiment(dataID + "_A")
            raceOK = True
        except:
            time.sleep(0.5)
            pass

    stats = {}
    for i, fExp in enumerate(fselExperiments):
        np.random.seed(i)
        random.seed(i)
        # for technical reasons we put the for loop here,
        # we now know which classifier we will use.
        # the feature selection still is applied on the whole
        for j, cExp in enumerate(clfExperiments):
            foldStats = {}
            foldStats["features"] = []
            foldStats["params"] = {}
            foldStats["params"].update(fExp)
            foldStats["params"].update(cExp)
            foldStats["params"].update({"Experiment": "A"})

            run_name = getRunID (foldStats["params"])
            current_experiment = dict(mlflow.get_experiment_by_name(dataID + "_A"))
            experiment_id = current_experiment['experiment_id']

            # check if we have that already
            # recompute using mlflow did not work, so i do my own.
            if len(glob (os.path.join(TrackingPath, str(experiment_id), "*/artifacts/" + run_name + ".ID"))) > 0:
                print ("X", end = '', flush = True)
                continue

            # log what we do now
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("A:(RUN) " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")

            fselector = createFSel (fExp)
            with np.errstate(divide='ignore',invalid='ignore'):
                fselector.fit (X, y)
            feature_idx = fselector.get_support()
            feature_names = X.columns[feature_idx]

            # apply selector-- now the data is numpy, not pandas, lost its names
            X_fs = fselector.transform (X)
            y_fs = y

            for k, (train_index, test_index) in enumerate(kfolds.split(X_fs, y_fs)):
                X_fs_train, X_fs_test = X_fs[train_index,:], X_fs[test_index,:]
                y_fs_train, y_fs_test = y_fs[train_index], y_fs[test_index]

                classifier = createClf (cExp)
                classifier.fit (X_fs_train, y_fs_train)
                y_pred = classifier.predict_proba (X_fs_test)[:,1]

                foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)
            foldStats["features"].append(list([feature_names][0].values))
            stats[str(i)+"_"+str(j)] = logMetrics (foldStats)
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("A: (DONE)" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")



    # experiment B
    raceOK = False
    while raceOK == False:
        try:
            mlflow.set_experiment(dataID + "_B")
            raceOK = True
        except:
            time.sleep(0.5)
            pass

    stats = {}
    for i, fExp in enumerate(fselExperiments):
        np.random.seed(i)
        random.seed(i)
        for j, cExp in enumerate(clfExperiments):
            foldStats = {}
            foldStats["features"] = []
            foldStats["params"] = {}
            foldStats["params"].update(fExp)
            foldStats["params"].update(cExp)
            foldStats["params"].update({"Experiment": "B"})
            run_name = getRunID (foldStats["params"])

            current_experiment = dict(mlflow.get_experiment_by_name(dataID + "_B"))
            experiment_id = current_experiment['experiment_id']

            # check if we have that already
            # recompute using mlflow did not work, so i do my own.
            if len(glob (os.path.join(TrackingPath, str(experiment_id), "*/artifacts/" + run_name + ".ID"))) > 0:
                print ("X", end = '', flush = True)
                continue

            # log what we do next
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("B:(RUN) " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")

            for k, (train_index, test_index) in enumerate(kfolds.split(X, y)):
                #print (test_index)
                X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
                y_train, y_test = y[train_index].copy(), y[test_index].copy()
                #X_train, X_test = X[train_index,:], X[test_index,:]
                #y_train, y_test = y[train_index], y[test_index]

                fselector = createFSel (fExp)
                with np.errstate(divide='ignore',invalid='ignore'):
                    fselector.fit (X_train.copy(), y_train.copy())
                feature_idx = fselector.get_support()
                feature_names = X_train.columns[feature_idx].copy()
                foldStats["features"].append(list([feature_names][0].values))

                # apply selector-- now the data is numpy, not pandas, lost its names
                X_fs_train = fselector.transform (X_train)
                y_fs_train = y_train

                X_fs_test = fselector.transform (X_test)
                y_fs_test = y_test

                # check if we have any features
                if X_fs_train.shape[1] > 0:
                    classifier = createClf (cExp)
                    classifier.fit (X_fs_train, y_fs_train)
                    y_pred = classifier.predict_proba (X_fs_test)
                    y_pred = y_pred[:,1]
                    foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)
                else:
                    # this is some kind of bug. if lasso does not select any feature and we have the constant
                    # classifier, then we cannot just put a zero there. else we get a different model than
                    # the constant predictor. we fix this by testing
                    if  cExp[0][0] == "Constant":
                        print ("F", end = '')
                        classifier = createClf (cExp)
                        classifier.fit (X_train.iloc[:,0:2], y_train)
                        y_pred = classifier.predict_proba (X_fs_test)[:,1]
                    else:
                        # else we can just take 0 as a prediction
                        y_pred = y_test*0 + 1
                    foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)

            stats[str(i)+"_"+str(j)] = logMetrics (foldStats)
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("B: (DONE)" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")




def executeExperiments (z):
    fselExperiments, clfExperiments, data, d = z
    executeExperiment ([fselExperiments], [clfExperiments], data, d)



if __name__ == "__main__":
    print ("Hi.")

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # load data first
    datasets = {}
    dList = ["Li2020", "Carvalho2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Ramella2018",   "Keek2020", "Park2020", "Song2020" , "Toivonen2019"]
    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        print ("\tLoaded data with shape", datasets[d].shape)
        # avoid race conditions later
        try:
            mlflow.set_tracking_uri(TrackingPath)
            mlflow.create_experiment(d+ "_A")
            mlflow.set_experiment(d + "_A")
            time.sleep(3)
        except:
            pass
        try:
            mlflow.set_tracking_uri(TrackingPath)
            mlflow.create_experiment(d+ "_B")
            mlflow.set_experiment(d + "_B")
            time.sleep(3)
        except:
            pass

    for d in dList:
        print ("\nExecuting", d)
        data = datasets[d]

        # generate all experiments
        fselExperiments = generateAllExperiments (fselParameters)
        print ("Created", len(fselExperiments), "feature selection parameter settings")
        clfExperiments = generateAllExperiments (clfParameters)
        print ("Created", len(clfExperiments), "classifier parameter settings")
        print ("Total", len(clfExperiments)*len(fselExperiments), "experiments")

        # generate list of experiment combinations
        clList = []
        for fe in fselExperiments:
            for clf in clfExperiments:
                clList.append( (fe, clf, data, d))

        # execute
        ncpus = 24
        fv = Parallel (n_jobs = ncpus)(delayed(executeExperiments)(c) for c in clList)

#
