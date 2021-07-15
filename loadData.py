import numpy as np
import os
import pandas as pd
from scipy.io import arff
import scipy.io as sio
from pprint import pprint
from sklearn.impute import SimpleImputer



# Define a class
class DataSet:
    def __init__(self, name):
        self.name = name

    def info (self):
        print("Dataset:", str(type(self).__name__), "\tDOI:", self.ID)

    def getData (self, folder):
        print("This octopus is " + self.color + ".")
        print(self.name + " is the octopus's name.")



class Song2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0237587"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0237587/")
        inputFile = "numeric_feature.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
        data["Target"] = np.asarray(data["label"] > 0.5, dtype = np.uint8)
        data = data.drop(["Unnamed: 0", "label"], axis = 1)
        return (data)



class Keek2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0232639"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0232639/Peritumoral-HN-Radiomics/")

        inputFile = "Clinical_DESIGN.csv"
        clDESIGNdata = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
        df = clDESIGNdata.copy()
        # remove those pats who did not die and have FU time less than 3 years
        df = clDESIGNdata[(clDESIGNdata["StatusDeath"].values == 1) | (clDESIGNdata["TimeToDeathOrLastFU"].values > 3*365)]
        target = df["TimeToDeathOrLastFU"] < 3*365
        target = np.asarray(target, dtype = np.uint8)

        inputFile = "Radiomics_DESIGN.csv"
        rDESIGNdata = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
        rDESIGNdata = rDESIGNdata.drop([z for z in rDESIGNdata.keys() if "General_" in z], axis = 1)
        rDESIGNdata = rDESIGNdata.loc[df.index]
        rDESIGNdata = rDESIGNdata.reset_index(drop = True)
        rDESIGNdata["Target"] = target

        # convert strings to float
        rDESIGNdata = rDESIGNdata.applymap(lambda x: float(str(x).replace(",", ".")))
        rDESIGNdata["Target"] = target

        return rDESIGNdata



class Li2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0227703"

    def getData (self, folder):
        # clinical description not needed
        # dataDir = os.path.join(folder, "journal.pone.0227703/")
        # inputFile = "pone.0227703.s011.xlsx"
        # targets = pd.read_excel(os.path.join(dataDir, inputFile))
        dataDir = os.path.join(folder, "journal.pone.0227703/")
        inputFile = "pone.0227703.s014.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        data["Target"] = data["Label"]
        data = data.drop(["Label"], axis = 1)
        return data



class Park2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0227315"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0227315/")
        inputFile = "pone.0227315.s003.xlsx"
        data = pd.read_excel(os.path.join(dataDir, inputFile), engine='openpyxl')
        target = data["pathological lateral LNM 0=no, 1=yes"]
        data = data.drop(["Patient No.", "pathological lateral LNM 0=no, 1=yes",
            "Sex 0=female, 1=male", "pathological central LNM 0=no, 1=yes"], axis = 1)
        data["Target"] = target
        return data



class Toivonen2019 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0217702"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0217702/")
        inputFile = "lesion_radiomics.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        data["Target"] = np.asarray(data["gleason_group"] > 0.0, dtype = np.uint8)
        data = data.drop(["gleason_group", "id"], axis = 1)
        return data



class Hosny2018A (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "HarvardRT.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        return data


class Hosny2018B (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "Maastro.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        return data


class Hosny2018C (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "Moffitt.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        return data



class Ramella2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0207455"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0207455/")
        inputFile = "pone.0207455.s001.arff"

        data = arff.loadarff(os.path.join(dataDir, inputFile))
        data = pd.DataFrame(data[0])
        data["Target"] = np.asarray(data['adaptive'], dtype = np.uint8)
        data = data.drop(['sesso', 'fumo', 'anni', 'T', 'N', "stadio", "istologia", "mutazione_EGFR", "mutazione_ALK", "adaptive"], axis = 1)
        return data



class Carvalho2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0192859"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0192859/")
        inputFile = "Radiomics.PET.features.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        # all patients that are lost to followup were at least followed for two
        # years. that means if we just binarize the followup time using two years
        # we get those who died or did not die within 2 years as binary label
        data["Target"] = (data["Survival"] < 2.0)*1
        data = data.drop(["Survival", "Status"], axis = 1)
        return data



if __name__ == "__main__":
    print ("Hi.")

    # small test
    # obtain data sets
    datasets = {}
    for d in [ "Carvalho2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Ramella2018",   "Toivonen2019", "Keek2020", "Li2020", "Park2020", "Song2020" ]:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        df = datasets[d]

    # stats
    for d in datasets:
        dimy = datasets[d].shape[0]/datasets[d].shape[1]
        b = np.round(100*(np.sum(datasets[d]["Target"])/len(datasets[d]) ))
        print (d, datasets[d].shape, dimy, b)
        print ("NAN:", datasets[d].isna().sum().sum())

#
