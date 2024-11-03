import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('indian_liver_patient.csv')

# Print the columns of the DataFrame (for debugging purposes)
print(data.columns)

# Convert categorical variables to numerical
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})  # Encode gender as 0 (Male) and 1 (Female)

# Prepare the feature and target variables
X = data.drop(['is_patient'], axis=1)  # Features: drop the target column
y = data['is_patient']  # Target variable (1 for liver disease, 0 for no disease)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the trained model to a file
joblib.dump(model, 'liver_disease_model.pkl')

# Feature importance visualization
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')

# Save the plot
plt.savefig('static/feature_importance.png')
plt.close()

print("Model saved as 'liver_disease_model.pkl'")
print("Feature importance plot saved as 'static/feature_importance.png'")
