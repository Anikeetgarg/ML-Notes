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

Computer vision libraries
torchvision - Contains datasets, models and image transformers often used for computer vision problems

torchvision.datasets  - example data sets

torchvision.models - already implemented vision models which can be incorporated in our model

torchvision.transforms -  transform image  to numerical data

torch.util.data.Dataset - base dataset class for PyTorch

torch.util.data.DataLoader - Creates a python iterable over dataset