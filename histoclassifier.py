import numpy as np
from scipy.io import loadmat

def histogram_transition_features(rr, window_size, bins=10):
    """
    Extract histogram-transition AF detection features.

    Features:
    1. Sum of squared histogram differences
    2. Number of nonempty bins
    3. Histogram height
    4. Standard deviation of ΔRR intervals
    """

    features = []

    # ΔRR intervals
    drr = np.diff(rr)

    for i in range(len(drr) - window_size + 1):

        window = drr[i:i + window_size]

        # Split window
        half = window_size // 2

        first_half = window[:half]
        last_half = window[half:]

        # Histograms
        hist1, _ = np.histogram(first_half, bins=bins, range=(-0.5, 0.5))
        hist2, _ = np.histogram(last_half, bins=bins, range=(-0.5, 0.5))

        # Normalize
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)

        ### Features ###
        # 1. Histogram SSD
        ssd = np.sum((hist1 - hist2) ** 2)

        combined_hist = hist1 + hist2

        # 2. Number of nonempty bins
        nonempty_bins = np.sum(combined_hist > 0)

        # 3. Histogram height
        hist_height = np.max(combined_hist)

        # 4. Standard deviation
        drr_std = np.std(window)

        feature_vector = [
            ssd,
            nonempty_bins,
            hist_height,
            drr_std
        ]

        features.append(feature_vector)

    return np.array(features)


def extract_labels(targets, window_size):
    """
    Label each window AF if majority of beats are AF.
    """

    labels = []

    # Match ΔRR length
    targets = targets[1:]

    # Slide a window over the RR-level labels
    for i in range(len(targets) - window_size + 1):
        window = targets[i:i + window_size]
        # Majority vote within the window
        labels.append(1 if np.mean(window) > 0.5 else 0)

    return np.array(labels)


def windows_to_rr_predictions(window_preds, signal_length, window_size):
    """
    Convert window-level predictions into RR-interval-level predictions
    using overlap voting.
    """
    votes = np.zeros(signal_length)
    counts = np.zeros(signal_length)

    # Distribute each window prediction back to the RR intervals it covers
    for i, pred in enumerate(window_preds):
        votes[i:i+window_size] += pred
        counts[i:i+window_size] += 1

    # Normalize by how many windows contributed to each RR interval
    avg_votes = votes / (counts + 1e-8)

    # Final RR-level decision
    detectRR = (avg_votes > 0.5).astype(int)

    return detectRR


def evaluate_performance(targetsRR, detectRR):
    """
    Evaluate AF detection performance at RR-interval level.

    Parameters:
    - targetsRR: ground truth labels per RR interval (0 = non-AF, 1 = AF)
    - detectRR: predicted labels per RR interval (0 = non-AF, 1 = AF)
    """

    TP = np.sum((targetsRR == 1) & (detectRR == 1))
    TN = np.sum((targetsRR == 0) & (detectRR == 0))
    FP = np.sum((targetsRR == 0) & (detectRR == 1))
    FN = np.sum((targetsRR == 1) & (detectRR == 0))

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)

    return sensitivity, specificity

# ================================================
# LOAD DATA
# ================================================
window_size = 30 # Rougthly how many beats we look at

patient_data = {}

X_train = []
y_train = []

print("\nLoading data...\n")

for patient in range(1, 8):
    file_path = f"AF_RR_intervals/afdb_{patient}.mat"
    data = loadmat(file_path)

    rr = data["rr"].squeeze()
    targetsRR = data["targetsRR"].squeeze()

    X = histogram_transition_features(rr, window_size)
    y = extract_labels(targetsRR, window_size)

    print(f"Patient {patient}: X={X.shape}, y={y.shape}")

    # Store features and targets for each patient
    patient_data[patient] = {"X": X, "targetsRR": targetsRR}

    ### Training patients ###
    if patient <= 4:
        X_train.append(X)
        y_train.append(y)


# Concatenate
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

print("\nTrain shape:", X_train.shape)

# ================================================
# Train model
# ================================================
weights = np.array([
    1.5,   # SSD
    1.0,   # nonempty bins
   -1.0,   # histogram height
    1.5    # ΔRR std
])

### Compute training scores ###
train_scores = X_train @ weights

# Threshold is chosen as midpoint between AF and non-AF
af_scores = train_scores[y_train == 1]
normal_scores = train_scores[y_train == 0]

threshold = np.mean([np.mean(af_scores), np.mean(normal_scores)])

print("\nThreshold:", threshold)

# ================================================
# Peformance validation
# ================================================
### TRAINING PERFORMANCE ###
print("\n================================================")
print("TRAINING PERFORMANCE")
print("================================================")

for patient in range(1, 5):
    X = patient_data[patient]["X"]
    targetsRR = patient_data[patient]["targetsRR"]

    scores = X @ weights
    window_predictions = (scores > threshold).astype(int)
    detectRR = windows_to_rr_predictions(window_predictions, len(targetsRR), window_size)
    sensitivity, specificity = evaluate_performance(targetsRR, detectRR)

    print(
        f"Patient {patient}: "
        f"Sensitivity={sensitivity:.3f}, "
        f"Specificity={specificity:.3f}"
    )

### TEST PERFORMANCE ###
print("\n================================================")
print("TEST PERFORMANCE")
print("================================================")

all_detectRR = []
all_targetsRR = []

for patient in range(5, 8):
    X = patient_data[patient]["X"]
    targetsRR = patient_data[patient]["targetsRR"]

    scores = X @ weights
    window_predictions = (scores > threshold).astype(int)
    detectRR = windows_to_rr_predictions(window_predictions, len(targetsRR), window_size)
    sensitivity, specificity = evaluate_performance(targetsRR, detectRR)

    print(
        f"Patient {patient}: "
        f"Sensitivity={sensitivity:.3f}, "
        f"Specificity={specificity:.3f}"
    )

    # Store per patient data
    all_detectRR.append(detectRR)
    all_targetsRR.append(targetsRR)


### OVERALL TEST PERFORMANCE ###
# Concatenate all RR predictions
all_detectRR = np.hstack(all_detectRR)
all_targetsRR = np.hstack(all_targetsRR)

# Final evaluation
overall_sens, overall_spec = evaluate_performance(
    all_targetsRR,
    all_detectRR
)

print("\n================================================")
print("OVERALL TEST PERFORMANCE")
print("================================================")

print(f"Sensitivity : {overall_sens:.3f}")
print(f"Specificity : {overall_spec:.3f}")