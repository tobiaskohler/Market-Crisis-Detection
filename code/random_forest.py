from data_handler import CSVHandler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *

#make background grey
sns.set_style('darkgrid')

features, labels = CSVHandler.csv_to_np(CSVHandler, filepath='../prepared_data/resampled.csv')
log_returns = features.iloc[:, 3].values

# Splitting data into training and testing sets

train_proportion = 0.5
test_proportion = 1 - train_proportion
training_period = int(train_proportion*len(log_returns))
testing_period = int(test_proportion*len(log_returns))
data_train = features[:training_period]
data_test = features[training_period:]
labels_train = labels[:training_period]
labels_test = labels[training_period:]


# csv each set to csv
data_train.to_csv('../prepared_data/data_train.csv')
data_test.to_csv('../prepared_data/data_test.csv')
labels_train.to_csv('../prepared_data/labels_train.csv')
labels_test.to_csv('../prepared_data/labels_test.csv')

print(f'Shape of training data: {data_train.shape}\nShape of testing data: {data_test.shape}\nShape of training labels: {labels_train.shape}\nShape of testing labels: {labels_test.shape}')

# Creating a random forest classifier
clf = RandomForestClassifier(n_estimators=10, n_jobs=-1) #using all cores
clf.fit(data_train, labels_train)

training_score = clf.score(data_train, labels_train)
test_score = clf.score(data_test, labels_test)

predictions = clf.predict(data_test)
print(f'Accuracy on training set: {training_score}')
print(f'Accuracy on test set: {test_score}')

print("#################")
print("### CLASSIFICATION REPORT ###")
classification_report = pd.DataFrame(classification_report(labels_test, predictions, output_dict=True))
print(classification_report)

print("### CONFUSION MATRIX ###")
confusion_matrix = multilabel_confusion_matrix(labels_test, predictions, labels=[-1, 0, 1])

for i in range(len(confusion_matrix)):
    if i == 0:
        cprint(f"### label -1 / RED ###", 'red') 
        true_negative = confusion_matrix[i][0][0]
        false_negative = confusion_matrix[i][1][0]
        true_positive = confusion_matrix[i][1][1]
        false_positive = confusion_matrix[i][0][1]
        cprint(f'True negative: {true_negative}\nFalse negative: {false_negative}\nTrue positive: {true_positive}\nFalse positive: {false_positive}', 'red')
        
           
    elif i == 1:
        cprint(f"### label 0 / YELLOW ###", 'yellow')
        true_negative = confusion_matrix[i][0][0]
        false_negative = confusion_matrix[i][1][0]
        true_positive = confusion_matrix[i][1][1]
        false_positive = confusion_matrix[i][0][1]
        cprint(f'True negative: {true_negative}\nFalse negative: {false_negative}\nTrue positive: {true_positive}\nFalse positive: {false_positive}', 'yellow')

    elif i == 2:
        cprint("### label 1 / GREEN ###", 'green')
        true_negative = confusion_matrix[i][0][0]
        false_negative = confusion_matrix[i][1][0]
        true_positive = confusion_matrix[i][1][1]
        false_positive = confusion_matrix[i][0][1]
        cprint(f'True negative: {true_negative}\nFalse negative: {false_negative}\nTrue positive: {true_positive}\nFalse positive: {false_positive}', 'green')


print("#################")



# FEATURE IMPORTANCE
# feature_importances = pd.Series(clf.feature_importances_, index=features.columns)
# feature_importances.nlargest(20).plot(kind='barh')
# plt.show()




# combine training and predictions

predictions = pd.DataFrame(predictions, columns=['predictions'])
predictions.index = data_test.index

original_with_predictions = pd.concat([features, predictions], axis=1)
plt.style.use='dark-background'

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(original_with_predictions['^GSPC'], color='black')
ax1.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['market_light']==-1, color='red', alpha=0.5)
ax1.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['market_light']==0, color='yellow', alpha=0.5)
ax1.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['market_light']==1, color='green', alpha=0.5)
ax1.set_title('Original data')

ax2.plot(original_with_predictions['^GSPC'], color='black')
ax2.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['predictions']==-1, color='red', alpha=0.5)
ax2.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['predictions']==0, color='yellow', alpha=0.5)
ax2.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['predictions']==1, color='green', alpha=0.5)
ax2.set_title('Predictions')

for i in range(len(original_with_predictions)-len(predictions), len(original_with_predictions)):
    if original_with_predictions['predictions'][i] != original_with_predictions['market_light'][i]:
        ax2.scatter(original_with_predictions.index[i], original_with_predictions['^GSPC'][i], marker='o', color='red')
        
original_with_predictions['difference'] = original_with_predictions['predictions'] - original_with_predictions['market_light']

s = 15
alpha = .5
for i in range(len(original_with_predictions)-len(predictions), len(original_with_predictions)):

    if original_with_predictions['difference'][i] != 0:
        
        if original_with_predictions['predictions'][i] == 1:
            ax3.scatter(original_with_predictions.index[i], original_with_predictions['market_light'][i], marker='o', color='black', s=s)
            ax3.scatter(original_with_predictions.index[i], original_with_predictions['predictions'][i], marker='^', color='blue', s=s,)
            
        elif original_with_predictions['predictions'][i] == -1:
            ax3.scatter(original_with_predictions.index[i], original_with_predictions['market_light'][i], marker='o', color='black', s=s)
            ax3.scatter(original_with_predictions.index[i], original_with_predictions['predictions'][i], marker='^', color='blue', s=s)
            
        elif original_with_predictions['predictions'][i] == 0:
            ax3.scatter(original_with_predictions.index[i], original_with_predictions['market_light'][i], marker='o', color='black', s=s,)
            ax3.scatter(original_with_predictions.index[i], original_with_predictions['predictions'][i], marker='^', color='blue', s=s)
        
    ax3.legend(['Original', 'Predictions'], loc='lower left')
    



ax1.axvline(x=data_train.index[-1], color='black', linestyle='--')
ax2.axvline(x=data_train.index[-1], color='black', linestyle='--')
plt.show()