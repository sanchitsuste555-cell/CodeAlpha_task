
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ================
# 1. LOAD DATASET
# ================

data = pd.read_csv("Iris.csv")

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ===============
# 2. CHECK DATA
# ===============

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

print("\nSpecies count:")
print(data["Species"].value_counts())


# ============================================
# 3. SELECT INPUT AND OUTPUT
# ============================================

# Input features
X = data[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

# Output/target
y = data["Species"]


# ============================================
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================
# 5. SCALE THE DATA
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================
# 6. CREATE KNN MODEL
# ============================================

model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)


# ============================================
# 7. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred)


# ============================================
# 8. CALCULATE ACCURACY
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy:", accuracy)


# ============================================
# 9. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# 10. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# 11. DISPLAY CONFUSION MATRIX
# ============================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Iris Flower Classification - Confusion Matrix")

plt.show()


# ============================================
# 12. VISUALIZE IRIS DATA
# ============================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=data,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Iris Flower Classification")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.show()