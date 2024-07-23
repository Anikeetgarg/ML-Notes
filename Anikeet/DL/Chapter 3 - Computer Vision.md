Computer vision - art of teaching computers to see

Steps in building CNN model

Helper function download
```python
import requests
from pathlib import Path
if Path("helper_functions.py").is_file():
    print("File already exits, skipping downloading")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)
```

0. Computer Vision library in pytorch
1. Load data
2. Prepare data
3. Model 0: Building a baseline model
4. Making prediction and evaluating model
5. Setup device agnostic for and evaluating on model 0
6. Model 1: Adding non linear layers
7. Model 2: Adding CNN layer
8. Comparing our models
9. Evaluating our best model
10. Making a confusion matrix
11. Saving and loading the best performing model


### 0. Computer Vision library in pytorch

Computer vision libraries
torchvision - Contains datasets, models and image transformers often used for computer vision problems

torchvision.datasets  - example data sets

torchvision.models - already implemented vision models which can be incorporated in our model

torchvision.transforms -  transform image  to numerical data

torch.util.data.Dataset - base dataset class for PyTorch

torch.util.data.DataLoader - Creates a python iterable over dataset


```python
# import basic torch
import torch 
from torch import nn

# import base libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import vision libraries
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.v2 import ToTensor # please change it, libraries are were being changed
```

### 1. Load data

```python
train_data = datasets.FashionMNIST(
    root= 'data', # where to down the data
    train= True, # do we want train data?
    download=True, # to download it locally?
    transform = ToTensor(), # how do we want to transform the data
    target_transform=None, # how do we want to transform the labels
)

test_data = datasets.FashionMNIST(
    root= 'data', # where to down the data
    train= False, # do we want train data?
    download=True, # to download it locally?
    transform = ToTensor(), # how do we want to transform the data
    target_transform=None, # how do we want to transform the labels
)

# Model output is in ordinal form, we need to convert it to text
class_names = train_data.classes

```

Visualize! Visualize! Visualize!

```python
# look at our 1st train sample
image, label = train_data[0]

image.shape, class_names[label]
#plt.imshow(image.squeeze())
#plt.title(class_names[label])
#or 
plt.imshow(image.squeeze(), cmap = "gray")
plt.title(class_names[label])
plt.axis(False)

```

Looking at random images from data_set
```python
torch.manual_seed(42)
fig = plt.figure(figsize =(9, 9))
rows, cols = 4, 4
for i in range(1, (rows * cols) + 1):
    rand_index = torch.randint(0, len(train_data), size = [1]).item()
    img, label = train_data[rand_index]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze())
    plt.title(class_names[label])
    plt.axis(False)
```

### Prepare data

torch.util.data.DataLoader - helps load data into models

in data loader we use mini-batches is more efficient, also our we might not be able to process 6 Million at one

```python
## Loading data

BATCH_SIZES = 32

from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset= train_data, 
                              batch_size= BATCH_SIZES,
                              shuffle=True,)

# dosent matter if data is shuffeled in test data, its easier for manually looking through when
# dataset is not shuffeled
test_dataloader = DataLoader(dataset= test_data, 
                              batch_size= BATCH_SIZES,
                              shuffle=False,)

print(train_dataloader, test_dataloader)
print("size ", len(train_dataloader), len(test_dataloader))

### checking shpae
train_feature_batch, train_label_batch  = next(iter(train_dataloader))
train_feature_batch.shape, train_label_batch.shape
```

Visualizing one img from data loader
```python
torch.manual_seed(42)

rand_idx = torch.randint(0, len(train_feature_batch), size= [1]).item()
img, label = train_feature_batch[rand_idx], train_label_batch[rand_idx]

plt.imshow(img.squeeze())
plt.title(class_names[label])
plt.axis(False)
plt.show()
```

### 3 Model 0: Building a baseline model

We build a base model (simple) and then improve upon it

**Note: We need to flatten our image currently its in 4 - D**

nn.Flatten() - helps with flattening

```python
# test how it works
flatten_model = nn.Flatten()

x = train_feature_batch[0]

output = flatten_model(x)

print(x.shape, output.shape)
```


Baseline model
```python
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape:int,
                hidden_units:int,
                 output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features= hidden_units),
            nn.Linear(in_features=hidden_units, out_features= output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)

# create an instance of the model 

torch.manual_seed(42)
model_0 = FashionMNISTModelV0(28*28, 10, len(class_names))
model_0
```

set up Loss function, accuracy function and optimizer

```python
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

from torchmetrics.classification import Accuracy

acc_fn = Accuracy(task = 'multiclass', num_classes = len(class_names))
```
We can also measure how much time it took for our model to train

```python
# machine learning is very experimental
# lets check how fast out model run
from timeit import default_timer as timer
def print_train_time(start:float, end:float, device: torch.device):
    total_time =   end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
    
start = timer()

end = timer()

print_train_time(start, end, "cpu")
```

Train / test loop
```python
torch.manual_seed(42)
from tqdm.auto import tqdm
train_time_on_cpu = timer()
epochs = 3
for epoch in range(epochs):
    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()

        y_pred = model_0.forward(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()
        if batch % 400 == 0:
            print(f'Look at {batch * len(X)} / {len(train_dataloader.dataset)} samples')

    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    model_0.eval()

    with torch.inference_mode():
        for batch_test , (X_test, y_test) in enumerate(test_dataloader):
            y_test_pred = model_0.forward(X_test)
            loss_test = loss_fn(y_test_pred, y_test)
            test_loss += loss_test

            test_acc = acc_fn(y_test, y_test_pred.argmax(dim = 1))

        test_loss /= len(test_dataloader)

    print(f'Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f} | Test acc {test_acc:.4f}')

train_time_on_cpu_end = timer()
total_train_time_with_model_0 = print_train_time(train_time_on_cpu, train_time_on_cpu_end, "cpu")
```

### 4 Making predictions and evaluating

```python
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn): 
    loss, acc = 0, 0
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
    
            # accumulate loss per batchabs
            loss += loss_fn(y_pred, y)
            acc += acc_fn(y, y_pred.argmax(dim = 1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name" : model.__class__.__name__,
            "model_loss" : loss,
            "model_acc"  : acc}

model_0_res = eval_model(model_0, test_dataloader, loss_fn, acc_fn)
```

### 5 Setup device agnostic for and evaluating on model 0

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

### 6 Model 1: Adding non linear layers

```python
class FashionMNISTModelV1(nn.Module):

	def __init__(self, input_features:int, hidden_units:int, output_features:int):
		super().__init__()
		self.layer_stack = nn.Sequential(
		nn.Flatten(),
		nn.Linear(in_features = input_features, out_features=hidden_units),
		nn.ReLU(),
		nn.Linear(in_features = hidden_units, out_features= output_features),
		nn.ReLU(),)

  

	def forward(self, x):
		return self.layer_stack(x)
		
model_1 = FashionMNISTModelV1(input_features= 28 * 28,
							hidden_units = 10,
							output_features = len(class_names)
							).to(device)

model_1.state_dict()
```
setting up loss, accuracy and optimizer
```python
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.1)

acc_fn = Accuracy(task = "MULTICLASS", num_classes=len(class_names)).to(device)
```

from now on we will have train and test steps

Train step 
```python
def train_step(model:nn.Module,
	data_loader: torch.utils.data.DataLoader,
	loss_fn: nn.Module,
	acc_fn,
	device: torch.device = device):
	
	train_loss, train_acc = 0, 0
	
	model.to(device)
	
	for batch, (X, y) in enumerate(data_loader):
	
		X = X.to(device)
		y = y.to(device)
		
		model.train()
		
		y_pred = model.forward(X)
		
		loss = loss_fn(y_pred, y)
		train_loss += loss
		train_acc = acc_fn(y_pred.argmax(dim = 1), y)
		
		optimizer.zero_grad()
		
		loss.backward()
		
		optimizer.step()
		
	train_loss /= len(data_loader)
	train_acc /= len(data_loader)
	
	print((f'Train loss {train_loss} | Train acc {train_acc}'))
```

test function

```python
def test_step(model:nn.Module,
	data_loader: torch.utils.data.DataLoader,
	loss_fn: nn.Module,
	acc_fn,
	device: torch.device = device
	):
	
	test_loss, test_acc = 0, 0
	model.to(device)
	
	with torch.inference_mode():
		model.eval()
		
		for X, y in data_loader:
			X = X.to(device)
			y = y.to(device)
			
			y_pred = model.forward(X)			  
			
			test_loss+= loss_fn(y_pred, y)
			test_acc += acc_fn(y_pred.argmax(dim = 1), y)
		
		test_loss /= len(data_loader)
		test_acc /= len(data_loader)

print(f"Test loss {test_loss} | Test acc {test_acc}")
```

Train loop 
```python 
epochs = 3
time_model_1_start = timer()

for epoch in tqdm(range(epochs)):

	train_step(model_1, train_dataloader, loss_fn, acc_fn, device)
	
	test_step(model_1, test_dataloader, loss_fn, acc_fn, device)
	
time_model_1_end = timer()

total_train_time_with_model_1 = time_model_1_end - time_model_1_start

print(total_train_time_with_model_1)
```