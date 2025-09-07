# Import all necessary libraries for data handling, preprocessing, modeling, and evaluation
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Step 1: Data Loading and Preparation ---
# Load the Iris dataset. It's a classic for a reason—it's clean and perfect for learning.
iris_data = load_iris()

# Create a DataFrame for features and a Series for the target variable.
# This makes the data easier to work with.
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.Series(iris_data.target)

# Split the dataset into training and testing sets.
# This is crucial for evaluating how well our model performs on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 2: Data Preprocessing (Scaling) ---
# Machine learning models often perform better when features are on a similar scale.
# StandardScaler adjusts the features so they have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data.
# We fit only on training data to prevent data leakage from the test set.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 3: Model Training ---
# Use Logistic Regression, a reliable choice for classification.
# The 'fit' method trains the model on our scaled training data.
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# --- Step 4: Model Evaluation ---
# Predict the species for the scaled test data.
y_pred = model.predict(X_test_scaled)

# Calculate and print key performance metrics.
# A high accuracy score means our model is making correct predictions most of the time.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris_data.target_names)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# --- Step 5: Model Persistence ---
# Saving the model is an important step in a real-world project.
# This allows us to use the trained model later without retraining it every time.
# We also save the scaler to ensure any new data is scaled identically before prediction.
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')

print("\nModel and Scaler saved successfully as 'iris_model.pkl' and 'iris_scaler.pkl'.")
