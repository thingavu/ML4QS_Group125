from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Load the data
data = pd.read_csv('./data_w_features/data_w_all_features_final_win10.csv')
data = data.dropna()

def ensemble_model(data, target_column='tone', drop_columns=['time_0.5', 'language', 'tone', 'participant', 'script'], train_size=0.8):
    # Define features and target
    X = data.drop(drop_columns, axis=1)
    y = data[target_column]

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=1)

    # Standardize features
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # Define classifiers
    clf1 = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = SVC(kernel='rbf', random_state=1, gamma=0.10, C=1.0)

    # Combine classifiers into ensemble model
    eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)], voting='hard')

    # Fit ensemble model
    eclf.fit(X_train_std, y_train)

    # Predict
    y_pred = eclf.predict(X_test_std)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", conf_matrix)

    # Print recall, precision, F1 score, accuracy
    print("Results on language:")
    print(classification_report(y_test, y_pred))

    return eclf, y_test, y_pred

eclf1, y1_test, y1_pred = ensemble_model(data, target_column='language')
eclf2, y2_test, y2_pred = ensemble_model(data, target_column='tone')

# Merge the two predictions
merged_label = []
merged_pred = []

for i in range(len(y1_test)):
    if y1_test[i] == 'ch' and y2_test[i] == 'bus':
        merged_label.append(1)
    elif y1_test[i] == 'ch' and y2_test[i] == 'casual':
        merged_label.append(2)
    elif y1_test[i] == 'en' and y2_test[i] == 'bus':
        merged_label.append(3)
    elif y1_test[i] == 'en' and y2_test[i] == 'casual':
        merged_label.append(4)
    
    if y1_pred[i] == 'ch' and y2_pred[i] == 'bus':
        merged_pred.append(1)
    elif y1_pred[i] == 'ch' and y2_pred[i] == 'casual':
        merged_pred.append(2)
    elif y1_pred[i] == 'en' and y2_pred[i] == 'bus':
        merged_pred.append(3)
    elif y1_pred[i] == 'en' and y2_pred[i] == 'casual':
        merged_pred.append(4)

# Confusion matrix
conf_matrix = confusion_matrix(merged_label, merged_pred)
print("Confusion matrix (merged):\n", conf_matrix)

# Print recall, precision, F1 score, accuracy
print("Results on merged:")
print(classification_report(merged_label, merged_pred))
