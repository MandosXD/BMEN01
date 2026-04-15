import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the MATLAB file
file_path = "AF_RR_intervals/afdb_2.mat"
data = loadmat(file_path)

for key in data.keys():
    value = data[key]
    print(f"{key}: type={type(value)}, shape={getattr(value, 'shape', None)}")

# Extract and clean
rr = data["rr"].squeeze()
qrs = data["qrs"].squeeze()
targets = data["targetsRR"].squeeze()
Fs = data["Fs"].squeeze()

print(data["afBounds"])
# Time axis (seconds)
time = np.cumsum(rr) / Fs

plt.figure(figsize=(12, 4))

# Plot RR intervals
plt.plot(time, rr, label="RR intervals", color="blue")

# Highlight AF regions
af_mask = targets == 1
plt.scatter(time[af_mask], rr[af_mask], color="red", s=10, label="AF")

plt.xlabel("Time (s)")
plt.ylabel("RR interval (samples)")
plt.title("RR Intervals with AF Episodes")
plt.legend()
plt.grid()

plt.show()

rr_norm = rr / np.mean(rr)

plt.hist(rr, bins=50)
plt.title("Distribution of RR intervals")
plt.show()