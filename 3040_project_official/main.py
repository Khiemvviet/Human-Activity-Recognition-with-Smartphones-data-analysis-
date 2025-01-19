import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)

# Define the entropy calculation function
def calculate_entropy(column):
    """
    Calculate the entropy of a column.
    """
    value_counts = column.value_counts(normalize=True)
    probabilities = value_counts.values
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Split the features into accelerometer and gyroscope based on column names
accelerometer_features = [col for col in df_train.columns if 'Acc' in col]
gyroscope_features = [col for col in df_train.columns if 'Gyro' in col]

# Calculate entropy for accelerometer and gyroscope features
accelerometer_entropies = {col: calculate_entropy(df_train[col]) for col in accelerometer_features}
gyroscope_entropies = {col: calculate_entropy(df_train[col]) for col in gyroscope_features}

# Sort and select the top 50 features based on lowest entropy
sorted_accelerometer_entropies = sorted(accelerometer_entropies.items(), key=lambda x: x[1])
sorted_gyroscope_entropies = sorted(gyroscope_entropies.items(), key=lambda x: x[1])

top_accelerometer_features = [feature for feature, _ in sorted_accelerometer_entropies[:50]]
top_gyroscope_features = [feature for feature, _ in sorted_gyroscope_entropies[:50]]

# Extract target variable 'Activity' from the train and test datasets
y_train = df_train['Activity']
y_test = df_test['Activity']

# Prepare accelerometer and gyroscope data
X_train_accel = df_train[top_accelerometer_features]
X_test_accel = df_test[top_accelerometer_features]
X_train_gyro = df_train[top_gyroscope_features]
X_test_gyro = df_test[top_gyroscope_features]

# Standardize the data
scaler = StandardScaler()
X_train_accel_scaled = scaler.fit_transform(X_train_accel)
X_test_accel_scaled = scaler.transform(X_test_accel)
X_train_gyro_scaled = scaler.fit_transform(X_train_gyro)
X_test_gyro_scaled = scaler.transform(X_test_gyro)

# Initialize classifiers
knn_accel = KNeighborsClassifier(n_neighbors=19)
knn_gyro = KNeighborsClassifier(n_neighbors=4)
dt_accel = DecisionTreeClassifier(random_state=42)
dt_gyro = DecisionTreeClassifier(random_state=42)
nb_accel = GaussianNB()
nb_gyro = GaussianNB()

# Train and predict for each classifier
classifiers = {
    "KNN (Accelerometer)": (knn_accel, X_train_accel_scaled, X_test_accel_scaled),
    "KNN (Gyroscope)": (knn_gyro, X_train_gyro_scaled, X_test_gyro_scaled),
    "Decision Tree (Accelerometer)": (dt_accel, X_train_accel_scaled, X_test_accel_scaled),
    "Decision Tree (Gyroscope)": (dt_gyro, X_train_gyro_scaled, X_test_gyro_scaled),
    "Naive Bayes (Accelerometer)": (nb_accel, X_train_accel_scaled, X_test_accel_scaled),
    "Naive Bayes (Gyroscope)": (nb_gyro, X_train_gyro_scaled, X_test_gyro_scaled),
}

results = {}
for name, (clf, X_train, X_test) in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=y_test.unique(), zero_division=0)
    results[name] = {"accuracy": acc, "precision": precision, "recall": recall, "confusion_matrix": cm, "report": report}

# Print classification reports and performance metrics
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro Recall: {metrics['recall']:.4f}")
    print(f"Classification Report:\n{metrics['report']}")

# Separate accelerometer and gyroscope results for plotting
accelerometer_results = {k: v for k, v in results.items() if "Accelerometer" in k}
gyroscope_results = {k: v for k, v in results.items() if "Gyroscope" in k}

fig, axes = plt.subplots(1, len(accelerometer_results), figsize=(20, 6))
fig.suptitle("Confusion Matrices for Accelerometer Data (with entropy)", fontsize=16)
for ax, (name, metrics) in zip(axes, accelerometer_results.items()):
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Oranges',
                xticklabels=y_test.unique(), yticklabels=y_test.unique(), ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot confusion matrices in subplots (gyroscope data) horizontally
fig, axes = plt.subplots(1, len(gyroscope_results), figsize=(20, 6))
fig.suptitle("Confusion Matrices for Gyroscope Data (with entropy)", fontsize=16)
for ax, (name, metrics) in zip(axes, gyroscope_results.items()):
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Oranges',
                xticklabels=y_test.unique(), yticklabels=y_test.unique(), ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()