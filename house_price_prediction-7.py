# House Price Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    "Age": [10, 8, 5, 7, 4, 6, 3, 2, 5, 1],
    "Price": [50, 60, 75, 85, 100, 110, 130, 145, 160, 190]
}

df = pd.DataFrame(data)

# Select input features and target
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display results
print("HOUSE PRICE PREDICTION")
print("----------------------")
print("Actual Prices:", y_test.values)
print("Predicted Prices:", y_pred.round(2))

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", round(r2, 2))

# Predict price for a new house
new_house = [[2000, 3, 2, 5]]
prediction = model.predict(new_house)

print("\nNew House Details:")
print("Area: 2000 sq.ft")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Age: 5 years")
print("Predicted Price:", round(prediction[0], 2), "Lakhs")
