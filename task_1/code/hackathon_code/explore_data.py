import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px


n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

def random_forest_exploring(X, y):
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid, n_iter=100,
                                   cv=3, verbose=2, random_state=42, n_jobs=-1)

    rf_random.fit(X, y)

    rf_random.best_params_# Creating importances_df dataframe
    importances_df = pd.DataFrame({"feature_names" : rf.feature_names_in_,
                                   "importances" : rf.feature_importances_})

    # Plotting bar chart, g is from graph
    g = px.bar(importances_df, x="feature_names", y="importances", title="Feature importances")
    g.show()