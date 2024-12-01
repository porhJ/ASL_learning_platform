import pandas as pd # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.metrics import confusion_matrix, classification_report # type: ignore
import pickle

# Load the dataset
data = pd.read_csv('hand_gesture_data.csv')

# Print unique labels to ensure multiple classes
print("Unique labels in dataset:", np.unique(data.iloc[:, -1].values))

# Split the data into features (X) and labels (y)
X = data.iloc[:, :-1].values  # Features: All columns except the last one (landmarks)
y = data.iloc[:, -1].values   # Labels: The last column (gesture labels)

# Check the unique labels in y
print("Unique labels in y before encoding:", np.unique(y))

# Encode the labels into numeric format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check the unique labels in y_encoded
print("Unique labels in y after encoding:", np.unique(y_encoded))

# Normalize the features (landmarks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling to your features

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Print classes in the training and test sets
print(f"Classes in y_train: {np.unique(y_train)}")
print(f"Classes in y_test: {np.unique(y_test)}")

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=3)
grid_search.fit(X_train, y_train)

# Print the best parameters from grid search
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Use the best model from the grid search
clf = grid_search.best_estimator_

# Evaluate the model using cross-validation
cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=3)
print(f"Cross-validated accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Fit the final model on the entire training set
clf.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print classification report
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model and label encoder to a file (pickle)
with open('gesture_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
    pickle.dump(label_encoder, model_file)  # Save label encoder as well
    pickle.dump(scaler, model_file)  # Save scaler for normalization during real-time recognition
