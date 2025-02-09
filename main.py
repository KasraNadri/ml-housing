from fetch import housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from util.ClusterSimilarity import ClusterSimilarity
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor


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

#= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
#================================= FINDING CORRELATION ==================================#
#= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

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

#= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
#================================= PREPARING THE DATA =================================#
#= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()


#========== STEP 1: CLEANING ==========#
imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include = [np.number])
imputer.fit(housing_num)


X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index)

#========== STEP 2: HANDLING CATEGORIAL ATTRIBUTES ==========#
housing_cat = housing[["ocean_proximity"]]
#print(housing_cat.head(10))

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot = housing_cat_1hot.toarray()


#========== STEP 3: FEATURE SCALING ==========#
min_max_scaler = MinMaxScaler(feature_range= (-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)


std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[['population']])


num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler()),
])

housing_num_prepared = num_pipeline.fit_transform(housing_num)

#numerical_attributes = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
#'total_bedrooms', 'population', 'households', 'median_income']
#categorial_attributes = ['ocean_proximity']

cat_pipeline = make_pipeline(
   SimpleImputer(strategy='most_frequent'),
   OneHotEncoder(handle_unknown='ignore')
)

#preprocessing = ColumnTransformer(
#    [
#        ('num', num_pipeline, numerical_attributes),
#        ('cat', cat_pipeline, categorial_attributes),
#    ]
#)

#housing_prepared = preprocessing.fit_transform(housing)


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10, gamma = 1., random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), 
                                     StandardScaler())


preprocessing = ColumnTransformer([
    ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']),
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population', 'households', 'median_income']),
    ('geo', cluster_simil, ['latitude', 'longitude']),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
],
remainder=default_num_pipeline) #===== ONE COLUMN REMAINING: HOUSING_MEDIAN_AGE

housing_prepared = preprocessing.fit_transform(housing)

lin_reg = make_pipeline(preprocessing, LinearRegression())

lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)


