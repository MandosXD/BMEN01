import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def histogram_features(rr, window_size, bins=10):
    features = []

    for i in range(len(rr) - window_size):
        window = rr[i:i+window_size]

        # Normalize window
        window = window / (np.mean(window) + 1e-8) # Avoid division by zero

        hist, _ = np.histogram(window, bins=bins, range=(0.5, 1.5))

        # Normalize histogram
        hist = hist / (hist.sum() + 1e-8) # Avoid division by zero
        features.append(hist)

    return np.array(features)


def extract_labels(targets, window_size):
    labels = []

    for i in range(len(targets) - window_size):
        window = targets[i:i+window_size]
        # If majority beats in window are AF then label window as AF
        labels.append(1 if np.mean(window) > 0.5 else 0)

    return np.array(labels)


# Load all datasets
X_train, y_train = [], []
X_test, y_test = [], []
window_size = 30 # Rougthly how many beats we look at

for n in range(1, 8):
    file_path = f"AF_RR_intervals/afdb_{n}.mat"
    data = loadmat(file_path)

    rr = data["rr"].squeeze()
    targets = data["targetsRR"].squeeze()

    X = histogram_features(rr, window_size)
    y = extract_labels(targets, window_size)

    print(f"Loaded afdb_{n}: X={X.shape}, y={y.shape}")

    # Train on patients 1-4, test on 5-7
    if n <= 4:
        X_train.append(X)
        y_train.append(y)
    else:
        X_test.append(X)
        y_test.append(y)

# Concatenate all patients
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "AF"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()