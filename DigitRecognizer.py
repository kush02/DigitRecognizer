from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

## Fetching MNIST data
mnist = fetch_mldata("MNIST original")

# There are 70,000 images (28 by 28 images for a dimensionality of 784)
#print(mnist.data.shape) # input
#print(mnist.target.shape) # target 
## Splitting dataset into train and test set and standardizing
from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
 mnist.data, mnist.target, test_size=1/7.0, random_state=42)


# Plotting first 5 images
plt.figure(figsize=(10,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

#################### Performing Scaling and PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_img)
train_imgS = scaler.transform(train_img)
test_imgS = scaler.transform(test_img)
pca = PCA(n_components=150,whiten=False).fit(train_imgS)
train_imgPCA = pca.transform(train_imgS)
test_imgPCA = pca.transform(test_imgS)

#################### Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Training model
t0=time()
gridC = {'C':[0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1,3,6]}
logisticRegr = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=0.01)
logisticRegr.fit(train_imgPCA, train_lbl)
predLogReg = logisticRegr.predict(test_imgPCA)

# Stats for logistic regression
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(test_lbl,predLogReg)
acc = cm.diagonal().sum()/float(len(test_lbl))
print("Accuracy of logistic regression: " + str(acc))
print(classification_report(test_lbl,predLogReg))
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True)
plt.ylabel('Actual label')
plt.title('Predicted label')
print("Time taken for script = " + str(time()-t0) + "s")
# Trying to do transfer learning for logistic regression so that misclassifieds are correctly classified
misClassified = []
totalMisClassified = []
for i in range(len(predLogReg)):
    if predLogReg[i] != test_lbl[i]:
        misClassified.append(predLogReg[i])
uniqueMisClassified = set(misClassified)
for digit in uniqueMisClassified:
    total = str(digit) + ":" + str(misClassified.count(digit))
    totalMisClassified.append(total)
print(totalMisClassified)
#boostLogReg = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=1.0)
"""
##################### SVM
from sklearn.svm import SVC
# Training model
t0 = time()
svm = SVC(C=10,kernel='rbf',class_weight='balanced') # check for optimal C and gamma
svm.fit(train_imgPCA,train_lbl)
predSVM = svm.predict(test_imgPCA)

# Stats for SVM
cm = confusion_matrix(test_lbl,predSVM)
acc = cm.diagonal().sum()/float(len(test_lbl))
print("Accuracy of SVM: " + str(acc))
print(classification_report(test_lbl,predSVM))
print("Time taken for script = " + str(time()-t0) + "s")

###################### Random Forest
from sklearn.ensemble import RandomForestClassifier as RFC
# Training model
t0 = time()
rndForest = RFC(n_estimators=100,min_samples_split=5,criterion='entropy',class_weight='balanced_subsample')
rndForest.fit(train_imgPCA,train_lbl)
predRF = rndForest.predict(test_imgPCA)

# Stats for Random Forest
cm = confusion_matrix(test_lbl,predRF)
acc = cm.diagonal().sum()/float(len(test_lbl))
print("Accuracy of Random Forest: " + str(acc))
print(classification_report(test_lbl,predRF))
print("Time taken for script = " + str(time()-t0) + "s")

##################### kNN
from sklearn.neighbors import KNeighborsClassifier as kNN
# Training model
t0 = time()
kN = kNN(n_neighbors=12,weights='distance')
kN.fit(train_imgPCA,train_lbl)
predkNN = kN.predict(test_imgPCA)

# Stats for kNN
cm = confusion_matrix(test_lbl,predkNN)
acc = cm.diagonal().sum()/float(len(test_lbl))
print("Accuracy of kNN: " + str(acc))
print(classification_report(test_lbl,predkNN))
print("Time taken for script = " + str(time()-t0) + "s")

#################### Neural Networks
from sklearn.neural_network import MLPClassifier as NN
# Training model
t0 = time()
network = NN(early_stopping=True,hidden_layer_sizes=(100,),alpha=1e-4)
network.fit(train_img,train_lbl)
predNN = network.predict(test_img)

# Stats for Neural Network
cm = confusion_matrix(test_lbl,predNN)
acc = cm.diagonal().sum()/float(len(test_lbl))
print("Accuracy of Neural Network: " + str(acc))
print(classification_report(test_lbl,predNN))
print("Time taken for script = " + str(time()-t0) + "s")

################### Stacked generalization
from sklearn.model_selection import ShuffleSplit
# Making training set
ss = ShuffleSplit(n_splits=6,test_size=1/7.)
ind = []
for trainInd,testInd in ss.split(mnist.data):
    ind.append(testInd) 
ind = np.array(ind).flatten()

# Scaling and PCA
trainSetS = scaler.transform(mnist.data[ind])
trainSetP = pca.transform(trainSetS)

# Making testing set
targetSet = mnist.target[ind]

# Training the model
t0 = time()
#corrArr = [predLogReg,predSVM,predRF,predkNN,predNN]
#print(np.corrcoef(corrArr))
stackTrain = [kN.predict(trainSetP),network.predict(trainSetP)]
stackTrain = np.array(stackTrain)
stackLabel = targetSet
stackLogReg = GridSearchCV(LogisticRegression(multi_class='multinomial',solver='lbfgs',warm_start=True),gridC)
stackLogReg.fit(stackTrain.T,stackLabel)
stackTest = [kN.predict(test_imgPCA),network.predict(test_imgPCA)]
stackTest = np.array(stackTest)
predStack = stackLogReg.predict(stackTest.T)

# Stats for Stacked Generalization
cm = confusion_matrix(test_lbl,predStack)
acc = cm.diagonal().sum()/float(len(test_lbl))
print("Accuracy of Stacked Generalization: " + str(acc))
print(classification_report(test_lbl,predStack))
print("Time taken for script = " + str(time()-t0) + "s")
"""
