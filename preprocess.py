import pandas as pd
# import torch


import warnings

warnings.filterwarnings("ignore")


class Preprocessing:
    def __init__(self):
        self.train = pd.read_csv("/home/ubuntu/Downloads/PulmonaryFibrosis/train.csv")
        self.test = pd.read_csv("/home/ubuntu/Downloads/PulmonaryFibrosis/test.csv")
        print("Total Recordings in Train Data: {:,}".format(len(self.train)))

    def head(self):
        df_train = self.train.head()
        df_test = self.test.head()
        print("Train", "\n", df_train)
        print("\n", "Test", "\n", df_test)

    def preprocess(self):
        print("Missing values: {} ".format(self.train.isnull().values.any()))
        print("{} unique patients in Train Data.".format(len(self.train["Patient"].unique())))
        # Recordings per Patient
        data = self.train.groupby(by="Patient")["Weeks"].count().reset_index(drop=False)
        # Sort by Weeks
        data = data.sort_values(['Weeks']).reset_index(drop=True)
        print("Min entries/week {}".format(data["Weeks"].min()), "\n" +
              "Max entries/week  {}".format(data["Weeks"].max()))
        # Select unique bio info for the patients
        data = self.train.groupby(by="Patient")[["Patient", "Age", "Sex", "SmokingStatus"]].first().reset_index(
            drop=True)
        print(data)


preprocessing = Preprocessing()
preprocessing.head()
