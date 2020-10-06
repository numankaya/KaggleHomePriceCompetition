import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

print(y.head())

# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF",
                 "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Review data
print(X.describe())

print(X.head())


# specify the model.
# For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

predictions = iowa_model.predict(X)
print(predictions)

# Review your result
home_data.head()
