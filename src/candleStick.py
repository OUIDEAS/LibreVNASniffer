import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = [
    (200, 0.022),
    (200, 0.066),
    (300, 0.052),
    (400, 0.023),
    (250, 0.061),
    (370, 0.023),
    (450, 0.028),
    (310, 0.030),
    (340, 0.014),
    (230, 0.108),
    (230, 0.083),
    (320, 0.014),
    (200, 0.044),
    (200, 0.041),
]
# Sample data (replace with your own)
np.random.seed(0)
distance = np.random.uniform(
    0, 400, 100
)  # Random floating point distances between 0mm and 400mm
sensitivity = (
    np.random.randn(100) * 10 + 50
)  # Random sensitivities, centered around 50 MHz/C

# Combine distance and sensitivity into a single list of tuples
data2 = list(zip(distance, sensitivity))

# Create a DataFrame from the list of tuples
df = pd.DataFrame(data, columns=["Distance", "Sensitivity"])

# Define the bins for the specific ranges you want
bins = [200, 300, 400, 500]

# Group by the defined distance intervals
df["Distance Group"] = pd.cut(df["Distance"], bins=bins, right=False)

# Convert the 'Distance Group' to a categorical column with ordered categories
df["Distance Group"] = pd.Categorical(
    df["Distance Group"],
    categories=pd.cut(bins, bins=bins, right=False).categories,
    ordered=True,
)

# Create the boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        df[df["Distance Group"] == group]["Sensitivity"]
        for group in df["Distance Group"].cat.categories
    ],
    labels=[str(group) for group in df["Distance Group"].cat.categories],
)
plt.title("Boxplot of Sensitivity by Distance Group")
plt.xlabel("Distance (mm) Group")
plt.ylabel("Sensitivity (MHz/°C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
