
print("starting")
from polyjuice import Polyjuice

print("success")
import pandas as pd
import numpy as np

# Load the dataset
feature_names = [
    "Age", "Workclass", "fnlwgt", "Education",
    "Education-Num", "Marital Status", "Occupation",
    "Relationship", "Race", "Sex", "Capital Gain",
    "Capital Loss", "Hours per week", "Country", "label"
]
df = pd.DataFrame(
   np.genfromtxt('adult.data', delimiter=', ', dtype=str),
   columns=feature_names
)
tabular_data = Tabular(
    df,
    categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
    target_column='label'
)