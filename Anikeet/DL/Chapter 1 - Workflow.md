### ML workflow 
1. Getting data ready
2. Building a model
	- Pick a loss function
	- Pick an optimizer
1. Fitting a training  data
2. Evaluating the model
3. Improving model through experimentation
4. Save and reload your model

 
### Data Preparing and Loading

Classic loading of data 
Visualise the data
split into train, test 

### Building a Model

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

### Fitting training data
training loop 
1. loop through data
2. Forward pass
3. Calculate the loss
4. Loss backwards
5. Optimizer step

```python
ephocs_count = []
train_loss_values = []
test_loss_values = []

ephocs = 500

for ephoc in range(ephocs):
    model_0.train() # pytorch will start tracking parameters and optimize for gradient decent

    # forward pass
    y_preds =  model_0.forward(X_train)

    # calculate the loss
    loss = loss_fn(y_preds, y_train)

    # reset the optimizer to zero
    optimizer.zero_grad()

    # loss backward 
    loss.backward()

    # steps the optimizer (perform the gradient decent)
    optimizer.step()

    # update the parameters
    model_0.eval()

    # stop parameter tracking
    with torch.inference_mode():
        # calcualte the loss on test set
        y_preds_ = model_0(X_test)

        # calculate lass
        loss_test = loss_fn(y_test, y_preds_)

    if ephoc % 10 == 0:
        print(f"Epochs = {ephoc} | Loss = {loss_test} | Params = {model_0.state_dict()}")
        ephocs_count.append(ephoc)
        train_loss_values.append(loss)
        test_loss_values.append(loss_test)
```

you can now plot how loss function decreases

```python
plt.plot(ephocs_count, np.array(torch.tensor(train_loss_values).numpy()), c = "orange", label = "Train loss")
plt.plot(ephocs_count, np.array(torch.tensor(test_loss_values).numpy()), c = 'blue', label = "Test loss")
plt.grid()
plt.legend()
```

Saving your model 

We are only saving the model params not the entire model 
*** its recommended to do so ***
torch.save() - saves the model, its params

torch.load() - only loads complete model, (can load attributes)

torch.nn.Module.load_state_dict() - used to load state params only 

```python 
# saving a model
from pathlib import Path

# creating a model_directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# creating model save path
MODEL_NAME = "my_model_01.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the models state_dict()
print(f"Saving to {MODEL_SAVE_PATH}")
torch.save(obj= model_0.state_dict(), f= MODEL_SAVE_PATH)

# loading a model

model__0 = LinearRegressionModel()
model__0.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
```