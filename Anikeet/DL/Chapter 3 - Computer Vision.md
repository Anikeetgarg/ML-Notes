Computer vision - art of teaching computers to see

Steps in building CNN model

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

