import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # зміна статі на циферки
        self.dataset.loc[self.dataset['Gender'] == 'F', 'Gender'] = 1
        self.dataset.loc[self.dataset['Gender'] == 'M', 'Gender'] = 0

        # зміна віку на циферки
        self.dataset.loc[self.dataset['Customer_Age'] >= 60, 'Customer_Age'] = 0
        self.dataset.loc[self.dataset['Customer_Age'] >= 40, 'Customer_Age'] = 0.5
        self.dataset.loc[self.dataset['Customer_Age'] >= 26, 'Customer_Age'] = 1

        le = LabelEncoder()

        le.fit(self.dataset['Income_Category'])
        self.dataset['Income_Category'] = le.transform(self.dataset['Income_Category'])

        columns_drop = ["CLIENTNUM", "Attrition_Flag", "Dependent_count", "Education_Level", "Marital_Status",
                        "Card_Category", "Months_on_book", "Total_Relationship_Count", "Credit_Limit",
                        "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
                        "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Bayes_1", "Bayes_2"]
        self.dataset = self.dataset.drop(columns_drop, axis=1)
        self.dataset = self.dataset.astype(int)
        self.dataset["Income_Category"] = self.dataset["Income_Category"] / self.dataset["Income_Category"].max()
        self.dataset["Contacts_Count_12_mon"] = self.dataset["Contacts_Count_12_mon"] / self.dataset[
            "Contacts_Count_12_mon"].max()
        self.dataset["Months_Inactive_12_mon"] = self.dataset["Months_Inactive_12_mon"] / self.dataset[
            "Months_Inactive_12_mon"].max()

        return self.dataset
