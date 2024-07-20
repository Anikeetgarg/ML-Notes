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
Different Data types
```python
float_32_tensor = torch.tensor([2., 3., 5.],
                                dtype = None, # what data type is used
                                device= None, # which device? cpu? gpu?
                               requires_grad = True)
```
Pytorch wants same datatype for variables and they should also run on same device
use `torch.cuda.FloatTensor` for device to run on gpu and `torch.FloatTensor` to run 32 bit float
We can convert data type from numpy to tensor

```python
p = np.array([1.2,2.2,3.2])
tensor_p = torch.from_numpy(p)

'''NOTE, converting from numpy to pytorch.tensor will convert it into 64 bit float. However, in torch by default 32-bit float is used so we usually convert it to 32 bit'''

tensor_p = tensor_p.to(torch.float32)
```

We can take transpose of a tensor
```python
TENSOR = torch.rand(2,3)
TENSOR
	tensor([[0.1023, 0.2694, 0.0585],
        [0.4099, 0.4583, 0.3281]])
TENSOR.T
	tensor([[0.1023, 0.4099],
        [0.2694, 0.4583],
        [0.0585, 0.3281]])
```
