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

Setting up loss function and optimiser

```python
## setting up a loss function

# loss_fn =nn.BCELoss() - This funciton assumes that the inputs are processed with sigmoid
loss_fn = nn.BCEWithLogitsLoss()

# this is equivalent to but its better in efficency
#loss_fn = nn.Sequential(
# nn.Sigmoid(),
# nn.BCELoss()
#)
optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.1)
```

### 3 Fitting the model to data (training)

```python
epoch_count = []

train_loss_val = []

test_loss_val = []

acc = []

  

epochs = 500

  

X_train, X_test = X_train.to(device), X_test.to(device)

y_train, y_test = y_train.to(device), y_test.to(device)

  

for epoch in range(epochs):
	model_1.train()
	y_preds = model_1.forward(X_train).squeeze()
	loss = loss_fn(y_preds, y_train)
	accuracy = accuracy_fn(y_true = y_train, y_pred = y_preds)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	model_1.eval()
	with torch.inference_mode():
		y_train_preds = model_1.forward(X_test)
		loss_test = loss_fn(y_train_preds.squeeze(), y_test)
		
	if epoch % 10 == 0:
		
		print(f"Epoch = {epoch} | accuracy = {accuracy} | train_loss = {loss} | test_loss = {loss_test}")
		
		epoch_count.append(epoch)
		
		acc.append(accuracy)
		
		train_loss_val.append(loss)
		
		test_loss_val.append(loss_test)
```
plotting loss

### 4 Making predictions and evaluating a model (inference)
```python
def plot_loss():

	plt.scatter(epoch_count, np.array(torch.tensor(train_loss_val).cpu().numpy()), label = 'Train', c="green")
	plt.scatter(epoch_count, np.array(torch.tensor(test_loss_val).cpu().numpy()), label = 'Test', c="orange")
	# plt.scatter(epoch_count, np.array(torch.tensor(acc).cpu().numpy()), label = 'Accuracy', c="lime")
	plt.title("Loss over time")
	plt.xlabel("Epoch")
	plt.ylabel("Loss Value")
	plt.grid()
	plt.legend()
```

make a helper function

```python
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
	print('helper_funciton already exits! skipping')
else:
	request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
	with open('helper_functions.py', 'wb') as f:
		f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
```

Plot the graph 
```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
```

### 5 Improving a model (from a model perspective)