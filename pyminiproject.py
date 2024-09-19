import pandas as pd 
import matplotlib as mtp 
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score

og_dataframe = pd.read_csv('C:\\salilpython\\miniproject\\diseasedataset_corrected.csv')

dataframe_total_X = og_dataframe[['Age', 'BMI', 'Blood Pressure', 'Cholesterol', 'Exercise Hours']]
dataframe_total_Y = og_dataframe[['Disease']]

dataframe_train_X = dataframe_total_X[0:61]
dataframe_train_Y = dataframe_total_Y[0:61]

dataframe_test_X = dataframe_total_X[61:]
dataframe_test_Y = dataframe_total_Y[61:]

scalar = StandardScaler()
scalar.fit_transform(dataframe_train_X)
scalar.transform(dataframe_test_X)


lin_model = LogisticRegression(max_iter=10000)
lin_model.fit(dataframe_train_X, dataframe_train_Y)
predicted_data = lin_model.predict(dataframe_test_X)
print(predicted_data)

coh_score = cohen_kappa_score(dataframe_test_Y, predicted_data)
print(coh_score)

