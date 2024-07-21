### ML workflow 
1. Getting data ready
2. Building a model
	- Pick a loss function
	- Pick an optimizer
1. Fitting a training  data
2. Evaluating the model
3. Improving model through experimentation
4. Save and reload your model

 
### 1 Data Preparing and Loading

Classic loading of data 
Visualise the data
split into train, test 

### 2 Building a Model

nn.Module - it is a base class that needs to be inherited when building a computation graph

nn.Parameter (type - tensor) -  Module (our models) parameters  are handled by pytorch

torch.optim contains various optimizer algorithms( these tell the model parameters how to best change in-order to improve gradient decent and in-turn reduce loss)

def forward() - A nn.Module subclass requires forward() method, it defines the computation that will take place when data is passed through the model
```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, 
										        dtype= torch.float,
										        requires_grad = True))
        self.bias = nn.Parameter(torch.randn(1, 
										    dtype = torch.float, 
										    requires_grad = True))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.weights * x + self.bias
```

How do i check my model parameters? 
```python 
model_0 = LinearRegressionModel()

# this method returns all the value of parameters
list(model_0.parameters()) # alone method returns a generator

# this method return all the param names and thier values
model_0.state_dict()
#list(model_0.state_dict()) use this to just get param names

```

How do i use test set to predict outputs?

**Note** torch.inference_mode() should always be used, it disables parameter tracking which is useful during training a model
```python
with torch.inference_mode():
	y_preds = model_0.forward(X_test)

y_preds
```
How do we evaluate our model? 
We need **loss function** ( evaluates how wrong your model is, the lower the better) 

A **optimiser** tells your model how to improve based on the loss function

```python
# this setup is ideal for regression

loss_fn = nn.L1Loss()

# lr = learning rate
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)

# for a classification problem we will use nn.BCELoss() - binary cross entropy
```