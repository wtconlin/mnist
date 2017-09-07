# # #  Will Conlin, 5/24/17

from IPython.display import display, HTML
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


train_df = pd.read_csv('./input/data70.csv', header = None)
test_df = pd.read_csv('./input/data30.csv', header = None)

# # # Format Data

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
validate_df = test_df.ix[:,0] # Validation

X_train = train_df.drop(train_df.columns[0], axis = 1)
Y_train = train_df.ix[:,0]
X_test  = test_df.drop(test_df.columns[0], axis = 1)
X_train.shape, Y_train.shape, X_test.shape

# # # Check data

# print ("xtrain\n")
# print (X_train.head())
# print ("ytrain\n")
# print (Y_train.head())
# print ("xtest\n")
# print (X_test.head())

# # # Trying different classifiers

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

# # # Confidence
# confidence = random_forest.predict_proba(X_test)[:, 1]

# output_confidence = pd.DataFrame({
        # "confidence": confidence
    # })
# output_confidence.to_csv('./output/confidence.csv', index=False)

# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print(acc_linear_svc)

# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print(acc_svc)

# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(acc_decision_tree)

# perceptron = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25,), random_state=1)
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print(acc_perceptron)

# # # Assign -1 to predictions with confidence ~0

size = validate_df.size
dropped = 0
max_dropped_10_percent = size * .1

# for i in range(size):
	# if dropped <= max_dropped_10_percent - 1:
		# if confidence[i] < 0.0001:
			# Y_pred[i] = -1
	# else:
		# break
	# dropped = dropped + 1

# print ("%d Samples Dropped (%f Percent)" %(dropped, dropped/size * 100))

# # # Validation

correct = 0
for i in range(size):
    if validate_df[i] == Y_pred[i] and Y_pred[i] != -1:
        correct = correct + 1
		
percent_correct = correct/(size-dropped) * 100

print("Validation: %f percent correct" %(percent_correct))

# # # Output

output = pd.DataFrame({
        "pr_y": Y_pred
    })
output.to_csv('./output/Conlin_predictions.csv', index=False)
