import pandas as pd
import numpy as np
df= pd.read_csv('backend/trainSql(1).csv')
df.head()
df['Label'].value_counts()
def find_var(feature_selected):
  variance_label_1 = df[df['Label'] == 1][feature_selected].var()
  variance_label_0 = df[df['Label'] == 0][feature_selected].var()
  print(f'Variance for feature : {feature_selected}')
  print(f'Variance for label 1: {variance_label_1}')
  print(f'Variance for label 0: {variance_label_0}')
  return variance_label_1-variance_label_0
no_var_list=[]
for features in df :
  var=find_var(features)
  print('\n')
  if var==0 :
    if features != 'Label' :
      no_var_list.append(features)

xx=['#:unix_secs', 'unix_nsecs', 'sysuptime', 'first', 'last','nexthop']
print(no_var_list)
no_var_list=no_var_list+xx
print(no_var_list)
df.drop(columns=no_var_list,inplace=True)
df.iloc[0]
df.head()
def normalize(df):
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized
nordf = normalize(df)
nordf.describe()
# import pandas as pd
# import matplotlib.pyplot as plt
label_counts = nordf['Label'].value_counts()
label_counts.plot(kind='bar')
# plt.xlabel('Label')
# plt.ylabel('Frequency')
# plt.title('Frequency of Labels')
# plt.show()
test_X = pd.read_csv('backend/testSql(3).csv')
test_Y = test_X['Label']
test_X.drop(['Label'], inplace=True, axis=1)
test_X.head()
test_nordf_Y= nordf['Label']
nordf.drop(['Label'], inplace=True, axis=1)
nordf.head()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(nordf, test_nordf_Y)
# y_pred = knn.predict(test_X)
# accuracy = accuracy_score(test_Y, y_pred)
# print("Accuracy:", accuracy)

# y_pred = knn.predict(test_X)

# # Calculate the confusion matrix
# cm = confusion_matrix(test_Y, y_pred)

# # Calculate the accuracy
# accuracy = accuracy_score(test_Y, y_pred)

# # Calculate the False Alarm Rate (FAR)
# far = cm[0, 1] / (cm[0, 1] + cm[0, 0])

# # Print the metrics
# print("Accuracy:", accuracy)
# print("FAR:", far)
# print("Confusion matrix",cm)
# prompt: now can you now give me a plot that calculates accuracy,far  for the abiove dataset while variyinh the number of neighboures

# import matplotlib.pyplot as plt

# # Define the range of neighbors to test
# neighbors = range(1, 20)

# # Initialize empty lists for storing accuracy and FAR values
# accuracy_values = []
# far_values = []

# # Loop through each number of neighbors
# for k in neighbors:
#   # Create a new KNN classifier with the current number of neighbors
#   knn = KNeighborsClassifier(n_neighbors=k)

#   # Fit the classifier on the training data
#   knn.fit(nordf, test_nordf_Y)

#   # Predict the labels for the test data
#   y_pred = knn.predict(test_X)

#   # Calculate the accuracy and FAR
#   accuracy = accuracy_score(test_Y, y_pred)
#   far = cm[0, 1] / (cm[0, 1] + cm[0, 0])

#   # Append the accuracy and FAR values to the lists
#   accuracy_values.append(accuracy)
#   far_values.append(far)


# print(accuracy_values)

# print(far_values)

# prompt: give me a code that plots there individually

# import matplotlib.pyplot as plt
# plt.plot(neighbors, accuracy_values)
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs. Number of Neighbors")
# plt.show()

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
    def calculatePCAComponents(self, X):
        pca = PCA()
        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        print(cumsum)
        d = np.argmax(cumsum >= 0.95) + 1
        plt.figure(figsize=(6,4))
        plt.plot(cumsum, linewidth=3)
        plt.axis([0, 10, 0, 1])
        plt.xlabel("Dimensions")
        plt.ylabel("Explained Variance")
        plt.plot([d, d], [0, 0.95], "k:")
        plt.plot([0, d], [0.95, 0.95], "k:")
        plt.plot(d, 0.95, "ko")
        plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
                arrowprops=dict(arrowstyle="->"), fontsize=16)
        plt.grid(True)
        plt.show()
        return d

    def reductionPCA_one(self, X, k, random_state_value ,whiten_value):
        pca = PCA(n_components=k, random_state=random_state_value, whiten=whiten_value)
        Y = pca.fit_transform(X)
        return Y

    def reductionPCA_auto(self, X, X_test, k, random_state_value ,whiten_value):
        pca = PCA(n_components=k, random_state=random_state_value, whiten=whiten_value)
        X = pca.fit_transform(X)
        X_test = pca.transform(X_test)
        return X, X_test

    def selectKBest(self, function, value, X, y):
        if function == "f_regression":
          return pd.DataFrame(SelectKBest(f_regression, k=value).fit_transform(X, y))
        if function == "chi2":
          print(SelectKBest(chi2, k=value).fit_transform(X, y))

    def extraTreeClassifier(self, X, y):
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        newX = model.transform(X)
        print (newX)
        return newX
fr = FeatureReductor()
red_trainX,red_testX = fr.reductionPCA_auto(nordf,test_X,9,45,True)
# knn = KNeighborsClassifier(n_neighbors=3)

#   # Fit the classifier on the training data
# knn.fit(red_trainX, test_nordf_Y)

#   # Predict the labels for the test data
# y_pred = knn.predict(red_testX)

  # Calculate the accuracy and FAR
# accuracy = accuracy_score(test_Y, y_pred)

# print(accuracy)
# fr.calculatePCAComponents(nordf)
# y_pred


class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


KNN = CustomKNN()
KNN.fit(red_trainX,test_nordf_Y)
KNN.predict(red_testX)

