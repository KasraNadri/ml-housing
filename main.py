from fetch import housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer



#print(housing.info())

#print(housing["ocean_proximity"].value_counts())
#print(housing['total_bedrooms'].value_counts())

#housing.hist(bins = 50, figsize = (12, 8))

#========== CREATING 5 LEVELS FOR INCOMES TO PREPARE IT FOR STRATIFIED SAMPLING ==========#
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

#========= STRATIFIED SAMPLING TO SPLIT THE DATA ==========#
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing['income_cat'])
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis = 1, inplace = True)

#========== COPYING THE STRATIFIED TRAINING SET TO BE WORKED WITH FROM NOW ON ==========#
housing = strat_train_set.copy()

#housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', grid = True, alpha = 0.3,
#             s = housing['population'] / 100, label = 'population', c = 'median_house_value', cmap = 'jet', colorbar = True,
#            legend = True, sharex = False, figsize = (10, 7))

#========== FINDING CORRELATION ==========#
corr_matrix = housing.drop('ocean_proximity', axis = 1).corr()
#print(corr_matrix['median_house_value'].sort_values(ascending = False))

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12, 8))
#housing.plot(kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha = 0.5, grid = True)
#plt.show()

housing['rooms_per_house'] = housing['total_rooms'] / housing['households']
housing['bedrooms_ratio'] = housing['total_bedrooms'] / housing['total_rooms']
housing['people_per_house'] = housing['population'] / housing['households']

corr_matrix = housing.drop('ocean_proximity', axis = 1).corr()
#print(corr_matrix['median_house_value'].sort_values(ascending = False))

#========== PREPARING THE DATA ==========#

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value".copy()]

#===== STEP 1: CLEANING =====#
imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include = [np.number])
imputer.fit(housing_num)


X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index)