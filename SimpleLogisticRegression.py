
# coding: utf-8

# In[105]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import classification_report,confusion_matrix

plt.style.use('seaborn-poster')

# from IPython import get_ipython
# get_ipython().magic('matplotlib inline')


# In[115]:

# Functions taken from pre-workshop materials 02_classification.ipynb, no changes made.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(10, 8)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# ### Import Data

# In[96]:

senator_data = pd.read_csv('SenateElectionHistory.tsv',sep='\t')    # import data to dataframe

#convert state names to integer IDs
state_to_int = dict(zip(set(senator_data['State']),[i for i in range(50)]))    
senator_data['State'] = [state_to_int[i] for i in senator_data['State']]

# Split into training and test data-sets

# pre-2015 = training set
training_data = senator_data.loc[senator_data['Year'] < 2015,]      
training_features = training_data.drop(['Democrat Winner','Democrat Margin'],inplace=False,axis=1)   #remove labels
training_labels = training_data['Democrat Winner']       #save labels to separate vector
print("Training data size: " + str(training_data.shape[0]))

# post-2015 = test set
testing_data = senator_data.loc[senator_data['Year'] >= 2015,]      
testing_features = testing_data.drop(['Democrat Winner','Democrat Margin'],inplace=False,axis=1)   #remove labels
testing_labels = testing_data['Democrat Winner']       #save labels to separate vector
print("Testing data size: " + str(testing_data.shape[0]))

print("\nFeatures: \n" + '\n'.join(senator_data.columns))


# ### Create Simple Logistic Regression Model
# Note: No tuning of parameters was performed.

# In[97]:

#fit logistic regression model
logreg = linear_model.LogisticRegression(C=1e5)    
logreg.fit(training_features, training_label)


# In[104]:

# predict results from the test data
predicted = logreg.predict(testing_features)

# plot the confusion matrix
cm = confusion_matrix(testing_labels,predicted)
plot_confusion_matrix(cm, classes=[0,1],
                      title='Confusion matrix, without normalization')


# ### Analysis of feature importances / "difficult-to-classify" cases

# In[106]:

# plot the feature weights, sorted by importance
feature_coefficients = pd.DataFrame.from_dict(dict(zip(testing_features.columns,logreg.coef_[0])), orient='index').sort_values(by=0)
feature_coefficients.plot(kind='barh', legend=False)
plt.title('Logistic Regression Feature Weights')
plt.show()


# In[114]:

# list the test data which was incorrectly predicted - both were predicted as '0' and should have been '1's
testing_features[[testing_labels[i] != predicted [i] for i in range(len(predicted))]]


# In[117]:

#TODO: could use predict_proba to see which predictions had stronger probabilities (?)
#logreg.predict_proba(testing_features)

