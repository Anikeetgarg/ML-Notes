What is a tensor?

a tensor is a multidimensional matrix

creation
```python 
scaler = torch.tensor(1) # 0 dimensional
vector = torch.tensor([1,2]) # 1 dimensional
MATRIX = torch.tensor([[1,2,3], [4,5,6]]) # 2 dimensional
TENSOR = torch.tensor([[[1,2,3], [1,2,3]], [[4,5,6],[4,5,6]]]) # 3 dimensional

# can also create random tensor 
dimension = (2,2,2) # 3 dimensional vector
TENSOR = torch.rand(dimension)
TENSOR = torch.rand(size = dimension)

# use 
TENSOR.ndim # check dimension of tensor

# To mask a tensor ( to tell model to not learn it )
# we creating tensors with all zeros or ones
TENSOR = torch.zeros(size = dimension)
TENSOR = torch.ones(size = dimension)
# to create tensors in a range we use
TENSOR = torch.arange(start= 0, end= 10, step= 1)

# if we want to create a tensor of sape say MATRIX we can create 
TENSOR_LIKE_MATRIX = torch.zeros_like(input = MATRIX)
TENSOR_LIKE_MATRIX = torch.ones_like(input = MATRIX)
TENSOR_LIKE_MATRIX = torch.rand_like(input = MATRIX)
```