# iris_classification_final.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Iris CSV file
csv_path = csv_path =csv_path = "C:/Users/shash/OneDrive/Desktop/Internship 2/task 1/Iris.csv"

df = pd.read_csv(csv_path)

# Step 2: Prepare features and target
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# Step 3: Encode target labels (Species → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

# Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 7: Visualize with pairplot
sns.pairplot(df, hue="Species")
plt.suptitle("🌸 Iris Dataset Visualization", y=1.02)
plt.show()
