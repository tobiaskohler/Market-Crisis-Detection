from data_handler import CSVHandler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *

features, labels = CSVHandler.csv_to_np(CSVHandler, filepath='../prepared_data/resampled.csv')
log_returns = features.iloc[:, 3].values

# Splitting data into training and testing sets

train_proportion = 0.7
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



# plot the most important features
feature_importances = pd.Series(clf.feature_importances_, index=features.columns)
feature_importances.nlargest(20).plot(kind='barh')
plt.show()




# trainings_score_list = []
# test_score_list = []
# iterations = 200

# for i in range(1, iterations):
#     clf = RandomForestClassifier(n_estimators=i)
#     clf.fit(data_train, labels_train)
#     training_score = clf.score(data_train, labels_train)
#     test_score = clf.score(data_test, labels_test)
#     trainings_score_list.append(training_score)
#     test_score_list.append(test_score)
    
#     print(f'Accuracy on training set: {training_score}')
#     print(f'Accuracy on test set: {test_score}')
    
# # plot the results
# plt.plot(range(1, iterations), trainings_score_list, label='Training score')
# plt.plot(range(1, iterations), test_score_list, label='Test score')
# plt.xticks(np.arange(1, iterations, 1))
# plt.xlabel('Number of trees')
# plt.ylabel('Model score')
# #mark the best score
# max_score = max(test_score_list)
# max_score_index = test_score_list.index(max_score)
# plt.scatter(max_score_index+1, max_score, c='r', label='Best score')

# plt.legend()
# plt.show()



