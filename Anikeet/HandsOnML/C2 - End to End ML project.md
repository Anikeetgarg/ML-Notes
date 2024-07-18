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

### 3 Explore and visualize the data to gain insights

