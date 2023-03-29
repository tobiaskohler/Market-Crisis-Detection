from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


'''

In this code, we define the number of folds for cross-validation (n_splits) and the number of trees in the random forest (n_estimators). We create a random forest classifier object (rfc) with the specified number of trees and a k-fold cross-validator object (kf) with the specified number of folds. We then loop over the folds and for each fold, we create the training and test sets using the split() method of the cross-validator object, train the random forest classifier on the training set, make predictions on the test set, and calculate the accuracy score for the current fold using the accuracy_score() function from scikit-learn. We add the score to a list of scores, and at the end of the loop, we calculate the mean and standard deviation of the accuracy scores.

Note that in this example, we assume that features and labels are numpy arrays containing the feature values and labels for the data, respectively.


'''


# Define the number of folds for cross-validation
n_splits = 5

# Define the number of trees in the random forest
n_estimators = 10

# Create a random forest classifier object
rfc = RandomForestClassifier(n_estimators=n_estimators)

# Create a k-fold cross-validator object
kf = KFold(n_splits=n_splits)

# Create an empty list to store the accuracy scores
scores = []

# Loop over the folds and train/evaluate the model on each fold
for train_index, test_index in kf.split(features):
    # Create the training and test sets for the current fold
    X_train = features[train_index]
    y_train = labels[train_index]
    X_test = features[test_index]
    y_test = labels[test_index]
    
    # Train the random forest classifier on the current fold
    rfc.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rfc.predict(X_test)
    
    # Calculate the accuracy score for the current fold
    score = accuracy_score(y_test, y_pred)
    
    # Add the score to the list of scores
    scores.append(score)

# Calculate the mean and standard deviation of the accuracy scores
mean_score = np.mean(scores)
std_score = np.std(scores)

print("Mean accuracy score: {:.2f}%".format(mean_score * 100))
print("Standard deviation of accuracy scores: {:.2f}%".format(std_score * 100))



