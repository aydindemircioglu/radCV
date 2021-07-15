#!/usr/bin/python3

import pandas as pd
import random
from scipy.stats import pearsonr
import scipy.stats as stats
from mlxtend.evaluate import ftest
import numpy as np
import time
import copy
import pickle
import shutil
import pathlib
import os
import math
import sys
import random
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
import tempfile


from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import colors

import roc_utils as ru
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.neural_network import MLPClassifier

from pymrmre import mrmr
from pprint import pprint
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV

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
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

from loadData import *
from utils import *


### parameters
TrackingPath = "/data/radCV/mlruns"



def testF1AUC (Y, scoresA, scoresB, verbose = True):
        bVals = []
        t, pA, pB = np.asarray(Y), np.asarray(scoresA), np.asarray(scoresB)
        for b in range(2000):
            np.random.seed(b)
            idx = np.random.choice(len(t), len(t))
            b_pA = pA[idx]
            b_pB = pB[idx]
            b_t = t[idx]
            b_fprA, b_tprA, b_thresholdsA = precision_recall_curve (b_t, b_pA)
            b_fprB, b_tprB, b_thresholdsB = precision_recall_curve (b_t, b_pB)
            area_under_curveA = auc (b_tprA, b_fprA)
            area_under_curveB = auc (b_tprB, b_fprB)
            bVals.append(area_under_curveA - area_under_curveB)

        fprA, tprA, thresholdsA = precision_recall_curve (Y, scoresA)
        area_under_curveA = auc (tprA, fprA)
        fprB, tprB, thresholdsB = precision_recall_curve (Y, scoresB)
        area_under_curveB = auc (tprB, fprB)

        z = np.sum(area_under_curveA -area_under_curveB)/np.std(bVals)
        p =2*scipy.stats.norm.cdf(-np.abs(z))
        return (p)




def  testAUC (Y, scoresA, scoresB, verbose = True):
        bVals = []
        t, pA, pB = np.asarray(Y), np.asarray(scoresA), np.asarray(scoresB)
        for b in range(2000):
            np.random.seed(b)
            idx = np.random.choice(len(t), len(t)) #list(range(len(t))), replace = True)
            b_pA = pA[idx]
            b_pB = pB[idx]
            b_t = t[idx]

            b_fprA, b_tprA, b_thresholdsA = roc_curve (b_t, b_pA)
            b_fprB, b_tprB, b_thresholdsB = roc_curve (b_t, b_pB)
            area_under_curveA = auc (b_fprA, b_tprA)
            area_under_curveB = auc (b_fprB, b_tprB)
            bVals.append(area_under_curveA - area_under_curveB)

        fprA, tprA, thresholdsA = roc_curve (Y, scoresA)
        area_under_curveA = auc (fprA, tprA)
        fprB, tprB, thresholdsB = roc_curve (Y, scoresB)
        area_under_curveB = auc (fprB, tprB)

        z = np.sum(area_under_curveA -area_under_curveB)/np.std(bVals)
        p =2*scipy.stats.norm.cdf(-np.abs(z))
        return (p)


def testAccuracy (Y, scoresA, scoresB, verbose = True):
        # bootstrap pvalue
        bVals = []
        t, pA, pB = np.asarray(Y), np.asarray(scoresA), np.asarray(scoresB)
        for b in range(2000):
            np.random.seed(b)
            idx = np.random.choice(len(t), len(t)) #list(range(len(t))), replace = True)
            b_pA = pA[idx]
            b_pB = pB[idx]
            b_t = t[idx]
            accA = accuracy_score (b_t, b_pA)
            accB = accuracy_score (b_t, b_pB)
            bVals.append(accA - accB)

        accA = accuracy_score (Y, scoresA)
        accB = accuracy_score (Y, scoresB)

        z = np.sum(accA -accB)/np.std(bVals)
        p =2*scipy.stats.norm.cdf(-np.abs(z))
        return (p)





def recreatePath (path, create = True):
        print ("Recreating path ", path)
        try:
                shutil.rmtree (path)
        except:
                pass

        if create == True:
            try:
                    os.makedirs (path)
            except:
                    pass
        print ("Done.")


if __name__ == "__main__":
    print ("Hi.")

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # load data first
    mlflow.set_tracking_uri(TrackingPath)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    results = {}
    dList = [ "Carvalho2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Ramella2018",   "Toivonen2019",
        "Keek2020", "Li2020", "Park2020", "Song2020" , ]
    # load data for later stats
    datasets = {}
    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")

    if os.path.exists("./results/results.feather") == False:
        for d in dList:
            for v in ["_A", "_B"]:
                current_experiment = dict(mlflow.get_experiment_by_name(d + v))
                experiment_id = current_experiment['experiment_id']
                runs = MlflowClient().search_runs(experiment_ids=experiment_id, max_results=50000)
                fTable = []
                for r in runs:
                    row = r.data.metrics
                    row["UUID"] = r.info.run_uuid
                    row["Model"] = r.data.tags["Version"]
                    row["Parameter"] = r.data.tags["pID"]

                    # compute also missing precision, recall
                    preds = pd.read_csv (os.path.join("/data/radCV/mlruns/",  str(experiment_id), str(r.info.run_uuid), "artifacts", "preds.csv" ))
                    precision, recall, thresholds = precision_recall_curve(preds["y_true"], preds["y_pred"])
                    y_pred_int = [int(k>=0.5) for k in preds["y_pred"]]
                    f1 = f1_score (preds["y_true"], y_pred_int)

                    precision = precision_score(preds["y_true"], y_pred_int, average='binary')
                    recall = recall_score(preds["y_true"], y_pred_int, average='binary')
                    row["Precision"] = precision
                    row["Recall"] = recall

                    fTable.append(row)
                results[d + v] = pd.DataFrame(fTable)
        print ("Pickling results")
        pickle.dump (results, open("./results/results.feather","wb"))
    else:
        print ("Restoring results")
        results = pickle.load(open("./results/results.feather", "rb"))

    print ("Shapes", [results[r].shape for r in results])


    # we can recreate AUC folds and compute +/- if need be
    # kfolds = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 1, random_state = 42)
    # d = "Li2020"
    # data = datasets[d]
    # y = data["Target"]
    # X = data.drop(["Target"], axis = 1)
    # for k, (train_index, test_index) in enumerate(kfolds.split(X, y)):
    #     print (test_index)




    # bring order into results
    for d in dList:
        pA = results[d+"_A"]["Parameter"]
        pA = [p.replace(", 'Experiment': 'A'", '') for p in pA]
        checkA = [p for p in pA if "Experiment" in p]
        if len(checkA) > 0:
            raise Exception ('Deleting exp key did go wrong.')

        pB = results[d+"_B"]["Parameter"]
        pB = [p.replace(", 'Experiment': 'B'", '') for p in pB]
        checkB = [p for p in pB if "Experiment" in p]
        if len(checkB) > 0:
            raise Exception ('Deleting exp key did go wrong.')


        res = [j for x in pA for j, y in enumerate(pB) if y == x]

        newpB = results[d+"_B"].iloc[res]["Parameter"]
        newpB = [p.replace(", 'Experiment': 'B'", '') for p in newpB]
        if newpB != pA:
            raise Exception ('Order NOT OK!')
        results[d+"_B"] = results[d+"_B"].iloc[res].reset_index(drop = True)


    # classifier/fsel combos
    fcImp = {}
    for m in ["AUC", "Sens", "Spec", "F1", "Precision", "Recall", "Accuracy"]:
        combos = results["Hosny2018A_A"]["Model"] .unique()

        fcImp[m] = {}
        for c in combos:
            fcImp[m][c] = []
        for d in dList:
            for c in combos:
                subA = results[d+"_A"][ results[d+"_A"]["Model"] == c].copy()
                subB = results[d+"_B"][ results[d+"_B"]["Model"] == c].copy()
                #fcImp[m][c].append ((subA[m] - subB[m]).values)
                # now take the best of those
                bestA = subA.sort_values(["AUC"], ascending = False).iloc[0]
                bestB = subB.sort_values(["AUC"], ascending = False).iloc[0]
                fcImp[m][c].append ((bestA[m] - bestB[m]))
            # print which is which
        for c in combos:
            if "Const" in c and "LASSO" in c:
                print(m, c, list(zip(dList,fcImp[m][c])))
        #fcImp[m][c] = [np.mean(z) for z in fcImp[m][c]]


    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    for m in ["AUC", "Sens", "Spec", "F1", "Precision", "Recall", "Accuracy"]:
        cpairs = [c.split("_") for c in combos]
        fsels = sorted(list(set(sorted(list(zip(*cpairs))[0]))))
        clfs = sorted(list(set(sorted(list(zip(*cpairs))[1]))))
        df = pd.DataFrame(index=clfs, columns=fsels)
        dfmin = pd.DataFrame(index=clfs, columns=fsels)
        dfmax = pd.DataFrame(index=clfs, columns=fsels)
        for f in fsels:
            for c in clfs:
                df.at[c,f] = np.mean(fcImp[m][f+"_" + c])
                dfmin.at[c,f] = np.min(fcImp[m][f+"_" + c])
                dfmax.at[c,f] = np.max(fcImp[m][f+"_" + c])

        dx = np.asarray(df.values, dtype = np.float64)
        fig, ax = plt.subplots(figsize = (10,10), dpi = 300)
        tnorm = colors.TwoSlopeNorm(vmin=-0.3, vcenter=0., vmax=0.10)
        ax.imshow(dx, cmap=plt.cm.bwr, norm = tnorm, interpolation='nearest')
        for i, c in enumerate(clfs):
            for j, f in enumerate(fsels):
                text = ax.text(j, i, r'\huge{' + str(np.round(df.at[c, f],3)) + "}\n" + "(" + str(np.round(dfmin.at[c, f],3)) + "-" + str(np.round(dfmax.at[c, f],3)) +  ")", ha="center", va="center", color="k", fontsize = 12)

        plt.yticks(list(range(len(clfs))), clfs, fontsize = 16)
        plt.xticks(list(range(len(fsels))), fsels, rotation = 45, fontsize = 16)
        plt.tight_layout()
        mStr = m
        if m == "AUC":
            mStr = "AUC-ROC"
        if m == "F1_AUC":
            mStr = "AUC-F1"
        ax.set_title("Importances for " + mStr, fontsize = 24)

        fig.savefig("./results/Importance_" + m + ".png", facecolor = 'w')
    plt.close('all')


    # get best _A vs _B_ for each
    plt.rc('text', usetex=False)
    allDiffs = {}
    rocPVals = {}
    f1PVals = {}
    accPVals = {}
    for d in dList:
        allDiffs[d] = {}
        for m in ["AUC", "Sens", "Spec", "F1", "Precision", "Recall", "Accuracy"]:
            diffs = results[d + "_A"][m] - results[d + "_B"][m]
            allDiffs[d][m] = diffs

        f, ax = plt.subplots(figsize = (10,10), dpi = 300)
        vals = allDiffs[d][m]
        vals = vals[int(0.05*len(vals)):int(0.95*len(vals))]
        plt.hist(vals, bins = np.arange(vals.min(), vals.max(), 0.01), color="#aaaaaa", edgecolor='k', alpha=0.65)
        plt.axvline(vals.mean(), color='r', linestyle='dashed', linewidth=1)
        plt.axvline(0, color='k', linestyle='dashed', linewidth=1)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.xlabel('1 - Specificity', fontsize = 22, labelpad = 12)
        # plt.ylabel('Sensitivity', fontsize= 25, labelpad = 12)

        ax.set_title("Difference per Model for " + m + " on " + d, fontsize = 24)
        f.savefig("./results/Hist_" + m + "_" + d + ".png", facecolor = 'w')
    plt.close('all')


    # get best _A vs _B_ for each, regardless of model
    bestModels= {}

    for d in dList:
        for v in ["_A", "_B"]:
            best = results[d + v].sort_values(["AUC"], ascending = False).iloc[0]
            print (d+v, best["AUC"])
            bestModels[d+v] = best.copy()

    # create AUC plots
    frow = 4

    pf, pax = plt.subplots(3, 4, figsize = (24, 18), dpi = 300)
    for w, d in enumerate(dList):
        # get preds for A and B
        current_experiment = dict(mlflow.get_experiment_by_name(d + "_A"))
        experiment_id = current_experiment['experiment_id']
        predsA = pd.read_csv (os.path.join("/data/radCV/mlruns/",  str(experiment_id), str(bestModels[d+"_A"]["UUID"]), "artifacts", "preds.csv" ))
        current_experiment = dict(mlflow.get_experiment_by_name(d + "_B"))
        experiment_id = current_experiment['experiment_id']
        predsB = pd.read_csv (os.path.join("/data/radCV/mlruns/",  str(experiment_id), str(bestModels[d+"_B"]["UUID"]), "artifacts", "preds.csv" ))

        from sklearn.utils import resample
        base_fpr = np.linspace(0, 1, 101)
        tprsA = []; tprsB = []
        aucsA = []; aucsB = []
        sensA = []; sensB = []
        specA = []; specB = []
        for b in range(2000):
            np.random.seed(b)
            random.seed(b)

            # bootstrap identifcal
            idx = resample(range(len(predsA["y_true"])))
            bs_tA, bs_pA = [predsA["y_true"][idx], predsA["y_pred"][idx]]
            bs_tB, bs_pB = [predsB["y_true"][idx], predsB["y_pred"][idx]]

            # curves
            fprA, tprA, thresholdsA = roc_curve (bs_tA, bs_pA)
            area_under_curveA = auc (fprA, tprA)
            aucsA.append(area_under_curveA)
            tpr = np.interp(base_fpr, fprA, tprA)
            tpr[0] = 0.0
            tprsA.append(tpr)
            sens, spec = findOptimalCutoff (fprA, tprA, thresholdsA)
            sensA.append(sens)
            specA.append(spec)

            fprB, tprB, thresholdsB = roc_curve (bs_tB, bs_pB)
            area_under_curveB = auc (fprB, tprB)
            aucsB.append(area_under_curveB)
            tpr = np.interp(base_fpr, fprB, tprB)
            tpr[0] = 0.0
            tprsB.append(tpr)
            sens, spec = findOptimalCutoff (fprB, tprB, thresholdsB)
            sensB.append(sens)
            specB.append(spec)


        fprA, tprA, thresholdsA = roc_curve (predsA["y_true"], predsA["y_pred"])
        area_under_curveA = auc (fprA, tprA)
        sens, spec = findOptimalCutoff (fprA, tprA, thresholdsA)

        print ("ORG A AUC/Sens/Spec:", area_under_curveA, sens, spec)
        print ("BOOT A AUC/Sens/Spec:", np.mean(aucsA), np.mean(sensA), np.mean(specA))

        fprB, tprB, thresholdsB = roc_curve (predsB["y_true"], predsB["y_pred"])
        area_under_curveB = auc (fprB, tprB)
        sens, spec = findOptimalCutoff (fprB, tprB, thresholdsB)

        print ("ORG B AUC/Sens/Spec:", area_under_curveB, sens, spec)
        print ("BOOT B AUC/Sens/Spec:", np.mean(aucsB), np.mean(sensB), np.mean(specB))





        assert sum(predsA["y_true"] - predsB["y_true"]) == 0
        Y = np.asarray(predsA["y_true"], dtype=np.uint8)
        scoresA = np.asarray(predsA["y_pred"])
        scoresB =  np.asarray(predsB["y_pred"])
        pval = testAUC(Y, scoresA, scoresB)
        rocPVals[d] = pval



        predsA_int = [int(k>=0.5) for k in predsA["y_pred"]]
        predsB_int = [int(k>=0.5) for k in predsB["y_pred"]]
        pval = testAccuracy(Y, predsA_int, predsB_int)
        accPVals[d] = pval

        pval = testF1AUC(Y, scoresA, scoresB)
        f1PVals[d] = pval



        tprsA = np.array(tprsA)
        mean_tprsA = tprsA.mean(axis=0)
        stdA = tprsA.std(axis=0)
        tprsA_upper = np.minimum(mean_tprsA + stdA, 1)
        tprsA_lower = np.maximum(mean_tprsA - stdA, 0)

        tprsB = np.array(tprsB)
        mean_tprsB = tprsB.mean(axis=0)
        stdB = tprsB.std(axis=0)
        tprsB_upper = np.minimum(mean_tprsB + stdB, 1)
        tprsB_lower = np.maximum(mean_tprsB - stdB, 0)


        f, ax = plt.subplots(figsize = (10,10), dpi = 300)
        plt.plot(base_fpr, mean_tprsA, 'r', label =  'AUC  (Incorrect): {0:0.2f}'.format(area_under_curveA) )
        plt.fill_between(base_fpr, tprsA_lower, tprsA_upper, color='red', alpha=0.15)

        plt.plot(base_fpr, mean_tprsB, 'b', label =  'AUC  (Correct): {0:0.2f}'.format(area_under_curveB) )
        plt.fill_between(base_fpr, tprsB_lower, tprsB_upper, color='blue', alpha=0.15)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('1 - Specificity', fontsize = 22, labelpad = 12)
        plt.ylabel('Sensitivity', fontsize= 25, labelpad = 12)

        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_aspect('equal', 'datalim')

        ax.set_title("AUC-ROC for " + d, fontsize = 30)
        ax.legend(loc="lower right", fontsize = 18)


        # add to big figure
        pax[w//frow][w%frow].plot(base_fpr, mean_tprsA, 'r', label =  'AUC  (Incorrect): {0:0.2f}'.format(area_under_curveA) )
        pax[w//frow][w%frow].fill_between(base_fpr, tprsA_lower, tprsA_upper, color='red', alpha=0.15)

        pax[w//frow][w%frow].plot(base_fpr, mean_tprsB, 'b', label =  'AUC  (Correct): {0:0.2f}'.format(area_under_curveB) )
        pax[w//frow][w%frow].fill_between(base_fpr, tprsB_lower, tprsB_upper, color='blue', alpha=0.15)

        pax[w//frow][w%frow].tick_params(axis = 'x' , labelsize=15)
        pax[w//frow][w%frow].tick_params(axis = 'y' , labelsize=15)
        pax[w//frow][w%frow].set_xlabel('1 - Specificity', fontdict= {'fontsize': 21})
        pax[w//frow][w%frow].set_ylabel('Sensitivity', fontdict= {'fontsize': 21})

        pax[w//frow][w%frow].plot([0, 1], [0, 1],'r--')
        pax[w//frow][w%frow].set_xlim([-0.01, 1.01])
        pax[w//frow][w%frow].set_ylim([-0.01, 1.01])
        pax[w//frow][w%frow].set_aspect('equal', 'datalim')

        pax[w//frow][w%frow].set_title(d, fontsize = 26)
        pax[w//frow][w%frow].legend(loc="lower right", fontsize = 16)

        f.savefig("./results/AUCROC_" + d + ".png", facecolor = 'w')



        # also compute P-R curves
        f, ax = plt.subplots(figsize = (10,10), dpi = 300)

        precision, recall, thresholds = precision_recall_curve(predsA["y_true"], predsA["y_pred"])
        area_under_curveA = auc (recall, precision)
        ax.plot(recall, precision, 'r', label =  'AUC-F1  (Incorrect): {0:0.2f}'.format(area_under_curveA) )

        precision, recall, thresholds = precision_recall_curve(predsB["y_true"], predsB["y_pred"])
        area_under_curveB = auc (recall, precision)
        ax.plot(recall, precision, 'b', label =  'AUC-F1  (Correct): {0:0.2f}'.format(area_under_curveB) )

        ax.plot([0, 1], [0, 1],'k--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Recall', fontsize = 22, labelpad = 12)
        plt.ylabel('Precision', fontsize= 25, labelpad = 12)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.set_aspect('equal', 'datalim')

        ax.set_title("Precision-Recall-Curve for " + d, fontsize = 27)
        ax.legend(loc="lower right", fontsize = 18)

        f.savefig("./results/PRC_" + d + ".png", facecolor = 'w')


    pf.tight_layout(pad=5.5)
    pf.delaxes(pax[2][3])
    pf.delaxes(pax[2][2])
    pf.savefig("./results/ROC_all.png", facecolor = 'w')
    plt.close('all')



    # table 2
    table2 = {}
    for d in dList:
        table2[d + "_A"] = {}
        table2[d + "_B"] = {}
        for m in ["AUC", "F1_AUC",  "Accuracy"]:
            table2[d + "_A"][m] = bestModels[d + "_A"][m]
            table2[d + "_B"][m] = bestModels[d + "_B"][m]
            diff = bestModels[d + "_A"][m] - bestModels[d + "_B"][m]
            table2[d + "_A"]["d_" +m] = diff
            table2[d + "_B"]["d_" + m] = diff
        table2[d+"_A"]["P_AUC"] = rocPVals[d]
        table2[d+"_A"]["P_F1"] = f1PVals[d]
        table2[d+"_A"]["P_Accuracy"] = accPVals[d]
        table2[d+"_B"]["P_AUC"] = 0
        table2[d+"_B"]["P_F1"] = 0
        table2[d+"_B"]["P_Accuracy"] = 0

    table2 = pd.DataFrame(table2).T
    # round
    table2 = table2.astype(float)
    table2 = table2.round(3)
    table2 = table2[['AUC', 'd_AUC',  'P_AUC',  'F1_AUC', 'd_F1_AUC',  'P_F1', 'Accuracy', 'd_Accuracy', 'P_Accuracy']]

    print(table2)
    table2.to_excel("./results/Table2.xlsx")



    # table 2
    table2 = {}
    for d in dList:
        table2[d] = {}
        for m in ["AUC", "Sens", "Spec", "F1", "Precision", "Recall", "Accuracy"]:
            diff = bestModels[d + "_A"][m] - bestModels[d + "_B"][m]
            table2[d][m] = diff
    table2 = pd.DataFrame(table2).T
    # round
    table2 = table2.astype(float)
    table2 = table2.round(3)
    print(table2)
    table2.to_excel("./results/TableS2.xlsx")



    # https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
        """Return an axes of confidence bands using a simple approach.

        Notes
        -----
        .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
        .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

        References
        ----------
        .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
           http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

        """
        if ax is None:
            ax = plt.gca()

        ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        ax.fill_between(x2, y2 + ci, y2 - ci, color="#111111", edgecolor="", alpha =0.15)

        return ax


    # Modeling with Numpy
    def equation(a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b)


    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    for m in ["AUC", "Sens", "Spec", "F1", "Precision", "Recall", "Accuracy"]:
        spfList = []
        for d in dList:
            spf =  datasets[d].shape[0]/datasets[d].shape[1]
            diff = bestModels[d + "_A"][m] - bestModels[d + "_B"][m]
            spfList.append ([spf, diff])
            print (d, spf, diff)

        R, pval = pearsonr(*zip (*spfList))
        R2 = R*R
        print (R, pval)

        x, y = zip(*spfList)
        p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
        y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

        # Statistics
        n =  len(x)                                          # number of observations
        ps = p.size                                                 # number of parameters
        dof = n - ps                                                # degrees of freedom
        t = stats.t.ppf(0.975, n - ps)                              # used for CI and PI bands

        # Estimates of Error in Data/Model
        resid = y - y_model
        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error


        # plot
        fig, ax = plt.subplots(figsize = (10,10), dpi = 300)
        ax.scatter (x,y, color = "k")
        ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=1.0, label="Fit")

        x2 = np.linspace(np.min(x), np.max(x), 100)
        y2 = equation(p, x2)

        # Confidence Interval (select one)
        plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
        #plot_ci_bootstrap(x, y, resid, ax=ax)

        # Prediction Interval
        pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
        # ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
        # ax.plot(x2, y2 + pi, "--", color="0.5")


        # Figure Modifications --------------------------------------------------------
        # Borders
        ax.spines["top"].set_color("0.5")
        ax.spines["bottom"].set_color("0.5")
        ax.spines["left"].set_color("0.5")
        ax.spines["right"].set_color("0.5")
        ax.get_xaxis().set_tick_params(direction="out")
        ax.get_yaxis().set_tick_params(direction="out")
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        # Labels
        #plt.title("Fit Plot for Weight", fontsize="14", fontweight="bold")

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('Bias', fontsize = 22, labelpad = 12)
        plt.xlabel('Samples/Features', fontsize= 22, labelpad = 12)
        plt.xlim(np.min(x)-0.05, np.max(x) + 0.05)
        #plt.ylim(np.min(y) - 0.025, np.max(y) + 0.025)

        right = 0.95
        ypos = 0.93
        legtext = ''
        if len(legtext ) > 0:
            ypos = 0.07
            legtext=legtext+"\n"

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        legpost = ''
        bbox_props = dict(fc="w", ec="0.5", alpha=0.9)
        pTxt = (' = {0:0.2f} ($p$ = {1:0.3f})').format(R2, pval)
        plt.text (right, ypos,
                  (legtext +  "$R^2$" + pTxt),
                  horizontalalignment='right',
                  size = 24, bbox  = bbox_props,
                  transform = ax.transAxes)
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})

        mStr = m
        if m == "AUC":
            mStr = "AUC-ROC"
        if m == "F1_AUC":
            mStr = "AUC-F1"
        ax.set_title("Bias vs Dimensionality (" + mStr + ")", fontsize = 28)
        print ("Bias for", m)
        fig.tight_layout()
        fig.savefig("./results/Bias_" + m + ".png", facecolor = 'w')

    plt.close('all')



#
