import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):

        self.dataset['Customer_Age'] = pd.cut(self.dataset['Customer_Age'], 5)
        # , "Attrition_Flag"
        columns_drop = ["CLIENTNUM", "Dependent_count", "Education_Level", "Marital_Status",
                        "Card_Category", "Months_on_book", "Total_Relationship_Count", "Credit_Limit",
                        "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
                        "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Bayes_1", "Bayes_2"]
        self.dataset = self.dataset.drop(columns_drop, axis=1)

        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        le.fit(self.dataset['Income_Category'])
        self.dataset['Income_Category'] = le.transform(self.dataset['Income_Category'])

        le.fit(self.dataset['Customer_Age'])
        self.dataset['Customer_Age'] = le.transform(self.dataset['Customer_Age'])

        self.dataset = self.dataset.astype(int)

        # Нормалізація данних
        self.dataset["Income_Category"] = (self.dataset["Income_Category"] - self.dataset["Income_Category"].min()) / \
                                          (self.dataset["Income_Category"].max() - self.dataset["Income_Category"].min())
        self.dataset["Contacts_Count_12_mon"] = (self.dataset["Contacts_Count_12_mon"] - self.dataset["Contacts_Count_12_mon"].min()) / \
                                                self.dataset["Contacts_Count_12_mon"].max() - self.dataset["Contacts_Count_12_mon"].min()
        self.dataset["Months_Inactive_12_mon"] = (self.dataset["Months_Inactive_12_mon"] - self.dataset["Months_Inactive_12_mon"].min()) / \
                                                 (self.dataset["Months_Inactive_12_mon"].max() - self.dataset["Months_Inactive_12_mon"].min())
        self.dataset["Customer_Age"] = (self.dataset["Customer_Age"] - self.dataset["Customer_Age"].min()) / \
                                       (self.dataset["Customer_Age"].max() - self.dataset["Customer_Age"].min())

        return self.dataset
