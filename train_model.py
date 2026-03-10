import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "pass": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["study_hours"]]
y = df["pass"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Prediction example
hours = [[5]]
prediction = model.predict(hours)

print("Prediction for 5 hours of study:", prediction)
