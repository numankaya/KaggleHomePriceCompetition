import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return mae

# Compare Different Tree Sizes


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y)
          for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

# make optimal size
final_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model
final_model.fit(X, y)
