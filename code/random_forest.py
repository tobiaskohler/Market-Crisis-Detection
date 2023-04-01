from data_handler import CSVHandler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *
import time

from hmm import calc_hmm

#make background grey
sns.set_style('darkgrid')

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
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1) #using all cores
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



## FEATURE IMPORTANCE
# feature_importances = pd.Series(clf.feature_importances_, index=features.columns)
# feature_importances.nlargest(20).plot(kind='barh')
# plt.show()




# combine training and predictions

predictions = pd.DataFrame(predictions, columns=['predictions'])
predictions.index = data_test.index

original_with_predictions = pd.concat([features, predictions], axis=1)
plt.style.use='dark-background'

hmm_df = calc_hmm()
original_with_predictions['market_light_hmm'] = hmm_df


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
ax1.plot(original_with_predictions['^GSPC'], color='black')
ax1.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['market_light']==-1, color='red', alpha=0.5)
ax1.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['market_light']==0, color='yellow', alpha=0.5)
ax1.fill_between(original_with_predictions.index, original_with_predictions['^GSPC'], where=original_with_predictions['market_light']==1, color='green', alpha=0.5)
ax1.set_title('Original data')

#add sma30 and sma200 to plot
original_with_predictions['sma_30'] = original_with_predictions['^GSPC'].rolling(window=30).mean()
original_with_predictions['sma_200'] = original_with_predictions['^GSPC'].rolling(window=200).mean()


#plot sma 30 and 200 on ax1 and make line thicker
ax1.plot(original_with_predictions['sma_30'], color='blue', linewidth=2)
ax1.plot(original_with_predictions['sma_200'], color='purple', linewidth=2)


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



# Calculate Performance of BAH vs Predictions
original_with_predictions['log_ret_adaptive'] = 0.0  # create a new column with initial values of 0.0
original_with_predictions['log_ret_bah'] = 0.0  # create a new column with initial values of 0.0

for i in range(len(original_with_predictions)):
    
    if not pd.isna(original_with_predictions['difference'][i]):
        original_with_predictions['log_ret_bah'][i] = original_with_predictions['log_ret'][i]
            
    else:
        original_with_predictions['log_ret_bah'][i] = 0.0
            
    
    if original_with_predictions['difference'][i] == 0:
        if original_with_predictions['market_light'][i] == 1:
            original_with_predictions['log_ret_adaptive'][i] = original_with_predictions['log_ret'][i]

        elif original_with_predictions['market_light'][i] == 0:
            original_with_predictions['log_ret_adaptive'][i] = 0.6 * original_with_predictions['log_ret'][i]

        elif original_with_predictions['market_light'][i] == -1:
            original_with_predictions['log_ret_adaptive'][i] = 0.0


    
    else:
        if original_with_predictions['predictions'][i] == 1:
            original_with_predictions['log_ret_adaptive'][i] = original_with_predictions['log_ret'][i]

        elif original_with_predictions['predictions'][i] == 0:
            original_with_predictions['log_ret_adaptive'][i] = 0.6 * original_with_predictions['log_ret'][i]

        elif original_with_predictions['predictions'][i] == -1:
            original_with_predictions['log_ret_adaptive'][i] = 0.0



#Compare to simple moving average strategy, if 30 day is below 200 day, no position, else long
original_with_predictions['sma_30_200'] = 0.0
original_with_predictions['log_ret_sma'] = 0.0

for i in range(len(original_with_predictions)):
    if not pd.isna(original_with_predictions['difference'][i]):

        if original_with_predictions['sma_30'][i] < original_with_predictions['sma_200'][i]:
            original_with_predictions['sma_30_200'][i] = 0.0
        else:
            original_with_predictions['sma_30_200'][i] = 1.0

        original_with_predictions['log_ret_sma'][i] = original_with_predictions['sma_30_200'][i] * original_with_predictions['log_ret'][i]
        
    else:
        original_with_predictions['log_ret_sma'][i] = 0.0
        
#### CHECK, IF RANDOM FOREST IS SUPERIOR TO SIMLE DRAWDOWN ADAPTIVE STRATEGY
original_with_predictions['market_light_by_drawdown'] = 0

for i in range(len(original_with_predictions)):
    if not pd.isna(original_with_predictions['difference'][i]):
        if original_with_predictions['drawdown'][i] <= -0.05:
            original_with_predictions['market_light_by_drawdown'][i] = -1
        elif original_with_predictions['drawdown'][i] > -0.05 and original_with_predictions['drawdown'][i] <= -0.01:
            original_with_predictions['market_light_by_drawdown'][i] = 0
        else:
            original_with_predictions['market_light_by_drawdown'][i] = 1
    else:
        original_with_predictions['market_light_by_drawdown'][i] = 0

original_with_predictions['log_ret_drawdown_adaptive'] = 0.0

for i in range(1, len(original_with_predictions)):
    
    if not pd.isna(original_with_predictions['difference'][i]):

        if original_with_predictions['market_light_by_drawdown'][i-2] == 1: # drawdown of 2 days ago
            original_with_predictions['log_ret_drawdown_adaptive'][i] = 1 * original_with_predictions['log_ret'][i]
            
        elif original_with_predictions['market_light_by_drawdown'][i-2] == 0: # drawdown of 2 days ago
            original_with_predictions['log_ret_drawdown_adaptive'][i] = 0.6 * original_with_predictions['log_ret'][i]
 
        else:
            original_with_predictions['log_ret_drawdown_adaptive'][i] = 0
            
    else:
        original_with_predictions['log_ret_drawdown_adaptive'][i] = 0.0
                    

#CHECK IF RANDOM FOREST IS SUPERIOR TO HIDDEN MARKOV MODEL
original_with_predictions['log_ret_hmm'] = 0.0

for i in range(1, len(original_with_predictions)):
    
    if not pd.isna(original_with_predictions['difference'][i]):

        if original_with_predictions['market_light_hmm'][i-2] == 1: # drawdown of 2 days ago
            original_with_predictions['log_ret_hmm'][i] = 1 * original_with_predictions['log_ret'][i]
            
        elif original_with_predictions['market_light_hmm'][i-2] == 0: # drawdown of 2 days ago
            original_with_predictions['log_ret_hmm'][i] = 0.6 * original_with_predictions['log_ret'][i]
 
        else:
            original_with_predictions['log_ret_hmm'][i] = 0
            
    else:
        original_with_predictions['log_ret_hmm'][i] = 0.0
        
    
ax4.plot(original_with_predictions['log_ret_bah'].cumsum(), color='blue')
ax4.plot(original_with_predictions['log_ret_adaptive'].cumsum(), color='red')
ax4.plot(original_with_predictions['log_ret_sma'].cumsum(), color='green')
ax4.plot(original_with_predictions['log_ret_drawdown_adaptive'].cumsum(), color='purple')
ax4.plot(original_with_predictions['log_ret_hmm'].cumsum(), color='cyan')


ax4.legend(['BAH', 'RF adaptive strategy', 'SMA(30/200 long only)', 'DD adaptive strategy', 'HMM'], loc='upper left')
ax4.set_ylabel('Cumulative return')
ax4.set_title('PERFORMANCE')



#calculate annualized sharpe ratio
sharpe_ratio_bah = np.sqrt(252) * (original_with_predictions['log_ret_bah'].mean() / original_with_predictions['log_ret_bah'].std())
sharpe_ratio_adaptive = np.sqrt(252) * (original_with_predictions['log_ret_adaptive'].mean() / original_with_predictions['log_ret_adaptive'].std())
sharpe_ratio_hmm = np.sqrt(252) * (original_with_predictions['log_ret_hmm'].mean() / original_with_predictions['log_ret_hmm'].std())


# add sharpe ratio to plot
ax4.text(0.5, 0.5, f'Sharpe ratio BAH: {sharpe_ratio_bah:.6f}', transform=ax4.transAxes)
ax4.text(0.5, 0.4, f'Sharpe ratio adaptive: {sharpe_ratio_adaptive:.6f}', transform=ax4.transAxes)
ax4.text(0.5, 0.3, f'Sharpe ratio HMM: {sharpe_ratio_hmm:.6f}', transform=ax4.transAxes)



# calculate drawdown of BAH vs. adaptive strategy
original_with_predictions['cum_ret_bah'] = original_with_predictions['log_ret_bah'].cumsum()
original_with_predictions['cum_ret_adaptive'] = original_with_predictions['log_ret_adaptive'].cumsum()
original_with_predictions['cum_ret_hmm'] = original_with_predictions['log_ret_hmm'].cumsum()

original_with_predictions['cum_max_bah'] = original_with_predictions['cum_ret_bah'].cummax()
original_with_predictions['drawdown_bah'] = original_with_predictions['cum_max_bah'] - original_with_predictions['cum_ret_bah']

original_with_predictions['cum_max_adaptive'] = original_with_predictions['cum_ret_adaptive'].cummax()
original_with_predictions['drawdown_adaptive'] = original_with_predictions['cum_max_adaptive'] - original_with_predictions['cum_ret_adaptive']

original_with_predictions['cum_max_hmm'] = original_with_predictions['cum_ret_hmm'].cummax()
original_with_predictions['drawdown_hmm'] = original_with_predictions['cum_max_hmm'] - original_with_predictions['cum_ret_hmm']


ax5.plot(original_with_predictions['drawdown_bah'], color='blue')
ax5.plot(original_with_predictions['drawdown_adaptive'], color='red')
ax5.plot(original_with_predictions['drawdown_hmm'], color='cyan')
ax5.legend(['BAH', 'Adaptive strategy', 'HMM Gausian'], loc='upper left')
ax5.set_ylabel('Drawdown')
ax5.set_title('DRAWDOWN')

#calculate sortino ratio
sortino_ratio_bah = np.sqrt(252) * (original_with_predictions['log_ret_bah'].mean() / original_with_predictions['log_ret_bah'][original_with_predictions['log_ret_bah'] < 0].std())
sortino_ratio_adaptive = np.sqrt(252) * (original_with_predictions['log_ret_adaptive'].mean() / original_with_predictions['log_ret_adaptive'][original_with_predictions['log_ret_adaptive'] < 0].std())
sortino_ratio_hmm = np.sqrt(252) * (original_with_predictions['log_ret_hmm'].mean() / original_with_predictions['log_ret_hmm'][original_with_predictions['log_ret_hmm'] < 0].std())


# add sortino ratio to plot
ax5.text(0.5, 0.5, f'Sortino ratio BAH: {sortino_ratio_bah:.6f}', transform=ax5.transAxes)
ax5.text(0.5, 0.4, f'Sortino ratio adaptive: {sortino_ratio_adaptive:.6f}', transform=ax5.transAxes)
ax5.text(0.5, 0.3, f'Sortino ratio HMM: {sortino_ratio_hmm:.6f}', transform=ax5.transAxes)


fig.set_size_inches(17.5, 9.5)
plt.show()


#save original with predictions to csv
original_with_predictions.to_csv('../predictions/original_with_predictions.csv')
plt.savefig(f'../predictions/BAH_vs_adaptive{time.time()}.png')
