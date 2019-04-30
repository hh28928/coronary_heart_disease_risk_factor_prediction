import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 700)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators}



def NN(train,test,pred):
    nn = MLPClassifier()
    nn.fit(train,pred)
    return nn.predict(test)


def RF(train,test,pred):
    rfc = RandomForestClassifier(n_estimators=400)
    rfc.fit(train,pred)
    return rfc.predict(test)



def kNN(train,test,pred):
    knn = KNeighborsClassifier(n_neighbors=300)
    knn.fit(train,pred)
    return knn.predict(test)


def tree(train,test,pred):
    dt = DecisionTreeClassifier()
    dt.fit(train,pred)
    return dt.predict(test)


data = pd.read_csv("cardio_train.csv", delimiter=';', engine='python')
pred = data['cardio']
cleandata = data.drop(data.columns[-1], axis=1)
cleandata = cleandata.drop(cleandata.columns[0], axis=1)  # Remove ID


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(cleandata,pred)

xTrain, xTest, yTrain, yTest = train_test_split(cleandata, pred, test_size=0.3,)


pca = PCA(n_components=8)
xTrain = pca.fit_transform(xTrain)
xTest = pca.transform(xTest)

scalar = StandardScaler()
xTrain = scalar.fit_transform(xTrain)
xTest = scalar.transform(xTest)


#i = 100
#while(i < 800):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(xTrain, yTrain)
#    output = knn.predict(xTest)
#    print(i,": ",f1_score(yTest,output,average='weighted'))
#    i = i + 100




temp = tree(xTrain,xTest,yTrain)

f1 = f1_score(yTest, temp, average='weighted')
print(f1)




#-----------PCA Variance--------#
#scaler = MinMaxScaler(feature_range=[0, 1])
#data_rescaled = scaler.fit_transform(cleandata)
#pca = PCA()
#scaled = pca.fit(data_rescaled)
#plt.figure()
##plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Cardiovascular Dataset Variance')
#plt.show()



#--------Grid Search for n_components------#
#random_rf = GridSearchCV(estimator=rfc,param_grid=random_grid,cv=5)
#random_rf.fit(train,pred)
#random_rf.best_params_



#------Feature score--------#

#cleandata = (cleandata - np.min(cleandata)) / (np.max(cleandata) - np.min(cleandata))

#test = SelectKBest(chi2,k='all')
#reduced = test.fit(cleandata,pred)
#np.set_printoptions()

#indices = np.argsort(reduced.scores_)
#features = []
#for i in range(11):
#    features.append(cleandata.columns[indices[i]])

#plt.figure()
#plt.bar(features, reduced.scores_[indices[range(11)]], color='r', align='center')
#plt.show()