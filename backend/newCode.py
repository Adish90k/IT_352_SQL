import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
df= pd.read_csv('backend/trainSql(1).csv')
class Preprocessor:

	def __init__(self):
		pass
	def adapt_dataset(self, df):
		labelColumn = df['Label']
		f_WithOut_Label = df.drop('Label', axis=1)
		columns_names = df.keys().drop('Label')
		return columns_names,labelColumn,f_WithOut_Label

	def maxScalerNormalization(self, X):
		scaler = MaxAbsScaler()
		scaler.fit(X)
		X_scaled = scaler.transform(X)
		return X_scaled

	def maxScalerNormalizationAuto(self, X, X_test):
		scaler = MaxAbsScaler()
		X_scaled = scaler.fit_transform(X)
		X_test_scaled = scaler.transform(X_test)
		return X_scaled, X_test_scaled

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class FeatureReductor:
    def __init__(self):
      pass

    def reductionPCA_one(self, X, k, random_state_value ,whiten_value):
        pca = PCA(n_components=k, random_state=random_state_value, whiten=whiten_value)
        Y = pca.fit_transform(X)
        return Y

    def reductionPCA_auto(self, X, X_test, k, random_state_value ,whiten_value):
        pca = PCA(n_components=k, random_state=random_state_value, whiten=whiten_value)
        X = pca.fit_transform(X)
        X_test = pca.transform(X_test)
        return X, X_test
def find_var(feature_selected):
  variance_label_1 = df[df['Label'] == 1][feature_selected].var()
  variance_label_0 = df[df['Label'] == 0][feature_selected].var()
  print(f'Variance for feature : {feature_selected}')
  print(f'Variance for label 1: {variance_label_1}')
  print(f'Variance for label 0: {variance_label_0}')
  variance_label_0 = df[df['Label'] == 0][feature_selected].var()
  return variance_label_1-variance_label_0
no_var_list=[]
for features in df :
  var=find_var(features)
  print('\n')
  if var==0 :
    if features != 'Label' :
      no_var_list.append(features)
df_test = pd.read_csv('backend/testSql(3).csv')
ls = ['#:unix_secs','unix_nsecs','sysuptime','last','nexthop','first']
df.drop(columns=no_var_list,inplace=True)
df.drop(columns=ls,inplace=True)
df_test.drop(columns=no_var_list,inplace=True)
df_test.drop(columns=ls,inplace=True)

pre = Preprocessor()
train_y = df.Label
df.drop(['Label'],inplace = True,axis = 1)
train_X = pre.maxScalerNormalization(df)
# coloumn_names,test_y,test_X = pre.adapt_dataset(df_test)
test_y = df_test.Label
df_test.drop(['Label'],inplace = True,axis = 1)
test_X = pre.maxScalerNormalization(df_test)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# class CustomKNN:
#     def __init__(self, n_neighbors=5):
#         self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

#     def fit(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X_test):
#         return self.model.predict(X_test)


# KNN = CustomKNN()
# KNN.fit(train_X,train_y)
# y_pred= KNN.predict(test_X)
# accuracy = accuracy_score(test_y, y_pred)
# print("Accuracy:", accuracy)
# KNN.predict()
import joblib

knn = KNeighborsClassifier()
knn.fit(train_X, train_y)

# Save the trained model to a file
joblib.dump(knn, 'knn_model.pkl')
