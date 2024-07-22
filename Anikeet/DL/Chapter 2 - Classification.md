A classification problem is involving prediction of where something is A or B (binary classification)

Types of classification:
- Binary Classification - Target can be one of two
- Multi-class Classification - Target can be one of many (> 2) entity
- Multi-label Classification - Target will have multiple labels

Steps to build a classification model 
0. Architecture of a classification neural network
1. Getting binary classification data ready
2. Building a PyTorch classification model
3. Fitting the model to data (training)
4. Making predictions and evaluating a model (inference)
5. Improving a model (from a model perspective)
6. Non-linearity
7. Replicating non-linear functions
8. Putting it all together with multi-class classification


### 0 Architecture of a classification neural network
General Architecture contains
- Input Layer - same number of features as data
- Hidden Layers - problem specific min = 1, max = inf
- Neurons per Hidden layer = 10 - 512 generally
- Output layer shape - 1 (for binary classification) or 1 per class for multi class classification
- Hidden layer activation - Usually ReLU but many others can be used
- Output activation - (torch.sigmoid) for binary classification (torch.softmax) for multiclass classification
- Loss function - BCELoss for binary classification or torch.nn.CrossEntropyLoss for multiclass
- Optimizer - SGD or Adam ( torch.optim.Optimizer)

### 1 Making classification Data and getting it ready
This is a toy dataset (data set)
```python
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples = n_samples, noise = 0.03, random_state= 42)

# convert to pd dataframe for ease of usage
import pandas as pd
df = pd.DataFrame({"X1": X[:, 0],
                   "X2": X[:, 1],
                  "label": y
                  })
```

Now we will visualize! visualize! visualize! 

```python
df.head()

def plot_predictions():
    plt.scatter(data = df, x ="X1", y = "X2", c = 'label')
    plt.title("Circles Visualization")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid()
    plt.legend()

plot_predictions()
```

Convert data into tensors
```python 
# check spare for consistency and getting an idea
X.shape, y.shape

# can use torch.from_numpy()
X_tensor = torch.tensor(X, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32)

X_tensor.shape, y_tensor.shape
```

Splitting into train and test data

```python 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor)
```

### 2 Building a PyTorch classification model
Manual model creation
```python
from torch import nn

class CircleModelV0(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer_1 = nn.Linear(in_features= 2, out_features= 5) # takes 2 featues and upsacles to 5 features
		self.layer_2 = nn.Linear(in_features= 5, out_features= 1)
		
	def forward(self, x: torch.tensor) -> torch.tensor:
		# x -> layer_1 -> layer_2
		return self.layer_2(self.layer_1(x))
```
Model using sequential 
```python
model_1 = nn.Sequential(

	nn.Linear(in_features = 2, out_features = 5),

	nn.Linear(in_features= 5, out_features= 1) ).to(device)

next(model_1.parameters()).device # to use gpu
```