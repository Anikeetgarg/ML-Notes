### Steps in developing End to End ML 
1. Look at the big picture
2. Get the data
3. Explore and visualize the data to gain insights
4. Prepare the data for machine learning algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solution
8. Launch, monitor, and maintain your system
#### 1 Looking at the big picture
- Frame the problem
		What is the business problem? Can it be solved by ML? What kind of ML? Supervised learning, Unsupervised learning? or other? what type of supervised/ unsupervised learning?
		Does the model need to be online? or batch learning would do fine?
- Select a performance Measure
		Choose a performance measure RMSE? MAE? or check competitions evaluation method tab
- Check assumptions
		Make a list of all the assumptions made by you or your colleges, this can help catch brutal errors early on
#### 2 Get the data
Load the data
```python
my_data = pd.read_csv('path')
```
take a quick look at data
```python
# first 5 rows
my_data.head()

# A description of each column
my_data.info()

# A description of values of each column
my_data.describe() # alternatively you can visualize it by 
my_data.hist(bins = 50, figsize = (12,8))

```
take a look at null values
```python
my_data.isnull().sum()
```

you can split data in train and test set
```python
# The problem with this is some data in train set might never bee seen by the model hence reducing accuracy of model on those values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y) 
```
One can also split numerical data into ranges

```python 
# splitting data in categories
house_data['income_cat']= pd.cut(house_data['median_income'],
                                 bins = [0., 1.5, 3.0, 4.5, 6.0, math.inf],
                                 labels=[1,2,3,4,5]
                                )

# selecting data from each category propotional to data values frequency in train set
strat_test_set, strat_train_set = train_test_split(house_data, stratify= house_data['income_cat])

# We can now just drop the colum after using it

for set_ in [train_set, test_set]:
    set_.drop(['income_cat'], axis=1, inplace=True)
```

^3b75bf

### 3 Explore and visualize the data to gain insights
- Visualizing geographical data
Look at longitude and latitude 
```python
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha= 0.2, grid = True)
plt.show()
```
Looking at the lat and long with respect to target variable
```python
housing.plot(kind = "scatter", x = 'longitude', y = 'latitude', 
             s = housing['population']/ 100,  # size of marker
             label = "population", 
             c = 'median_house_value', # color of marker
             # cmap = "jet", - dosent work for some reason
             colorbar = True,
             sharex= False, 
             figsize = (10, 7)
            )
plt.show

```
- Looking for co relations
```python
temp = housing.copy()
temp.drop(['ocean_proximity'], axis = 1, inplace=True) # droping categorical colums
corr_matrix = temp.corr()
corr_matrix.median_house_value.sort_values(ascending=False)
```
	median_house_value    1.000000
	median_income         0.689651
	total_rooms           0.131511
	housing_median_age    0.105997
	households            0.062816
	total_bedrooms        0.046362
	population           -0.028366
	longitude            -0.043639
	latitude             -0.146462
1 means very positively co-related, -1 means very negatively co-related, 0 means show no linear co relation
*Note = You can use feature engineering to create new features here*

### 4 Prepare the data for machine learning algorithms
- Input Value transformation
Most machine learning algorithms cannot work with missing features. So we need to clean the data

We can do the following with the missing data

```python 
# drop the entire column
housing.dropna(subset=["total_bedrooms"], inplace=True) # option 1
# drop just the rows having missing data
housing.drop("total_bedrooms", axis=1) # option 2

# or fill the missing data with some value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median') # this only work with numerial data

'''
We can use various values for strategy,
Numerical data
- mean - fills value using mean
Categorical data
- most_frequent
- constant , and another variable fill_value can be set to fill it with constants
'''

# Some other types of imputers are 
# KNNimputer - each value is filled with its neighourest neighbour value
# IterativeImputer - trains a regression model per feature to predict the missing value based on all other variables 
```

Categorical features transformation
```python
# used for non related categories
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# used for related categories
# e.g. best > okay > bad
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
```

Feature Scaling
ML algorithms do not perform well when the scales of data is different
we can scale using two techniques, min-max scaling, standardisation
```python 
# minmax scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# standardisation
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```
If scale is very wide and values in extremes are not very rare ( there is a even or a very good number of features at extremes) - scaling with squish data because of large scale 

here we can take **square-root** of the numbers if features are in not a very big zone , 
but if its something like power law distribution we can apply **log** to the whole data

we can also use the bucketing technique like [[C2 - End to End ML project#^3b75bf]]

Sometime out data has 2 peaks we can add transformational multimodial distribution for each of the peak. Each feature that depends on closeness to the peaks will benefit
```python
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
# can add multiple 
# age_simil_40 = rbf_kernel(housing[["housing_median_age"]], [[40]], gamma=0.1)
```

- Target Value Transformation
If we transform target variable (say take log) then the output will also be in log to make out life easy Sklearn has TransformedTargetRegressor which converts value before training and convert value for predict in to log and return the value out of log ( for out specific example)
```python
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(LinearRegression(),
									transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
```

#### Custom column transformers
Sometimes we will need to convert values into log or apply custom transformation, fortunately we can create our own transformers
```python
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# you can pass in arguments
rbf_transformer = FunctionTransformer(rbf_kernel,
										kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# you can pass in multi dimensional array 
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y = [sf_coords],   gamma = 0.1))
sf_simil = sf_transformer.fit_transform(housing[['latitude', 'longitude']])

# you can also perform feature engineering
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
```

to make custom transformers and make them able to learn i.e. fit() and later use transform()
we need only 3 methods to do so fit() - always returns self, transform() and fit_transform()
an example- standard scaler implementation as a custom funciton

```python 
from sklearn.base import BaseEstimator, TransformerMixin # - we need this as base
from sklearn.utils.validation import check_array, check_is_fitted # we need this for validation of data
class StandardScalerClone(BaseEstimator, TransformerMixin):
	def __init__(self, with_mean=True): # no *args or **kwargs!
		self.with_mean = with_mean
	def fit(self, X, y=None): # y is required even though we don't use it
		X = check_array(X) # checks that X is an array with finite float values
		self.mean_ = X.mean(axis=0)
		self.scale_ = X.std(axis=0)
		self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
		return self # always return self!
	def transform(self, X):
		check_is_fitted(self) # looks for learned attributes (with trailing _)
		X = check_array(X)
		assert self.n_features_in_ == X.shape[1]
		if self.with_mean:
			X = X - self.mean_
		return X / self.scale_
```
- All Scikit-Learn estimators set n_features_in_ in the fit() method, and they•
ensure that the data passed to transform() or predict() has this number of
features.
- This implementation is not 100% complete: all estimators should set
feature_names_in_ in the fit() method when they are passed a DataFrame.
Moreover, all transformers should provide a get_feature_names_out() method,
as well as an inverse_transform() method when their transformation can be
reversed. See the last exercise at the end of this chapter for more details.

```python 
from sklearn.cluster import KMeans
class ClusterSimilarity(BaseEstimator, TransformerMixin):
	def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
		self.n_clusters = n_clusters
		self.gamma = gamma
		self.random_state = random_state
	def fit(self, X, y=None, sample_weight=None):
		self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
		self.kmeans_.fit(X, sample_weight=sample_weight)
		return self # always return self!
	def transform(self, X):
		return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
	def get_feature_names_out(self, names=None):
		return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```