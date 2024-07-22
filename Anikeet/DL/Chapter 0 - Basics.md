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
we can multiply tensor
```python
TENSOR_A = torch.tensor([[1,2],
                        [3,4],
                        [5,6]])

TENSOR_B = torch.tensor([[7,10],
                         [8,11],
                         [9,12]])
# note here the dimensions are same! so we take transpose
torch.mm(TENSOR_A, TENSOR_B.T)
```
min, max, mean
```python
x = torch.arange(0, 100, 10)

x.min(), x.max()

# note - we cannot take min of int64 we need to convert it to relevent data type here we convert to float32
x = x.to(torch.float32).mean()
```
location of min and location of max
```python
x.argmin()
x.argmax()
```
change the shape of tensor
```python
x = torch.arange(1, 10) # 1 dimension tensor
x.reshape(1, 9) # 2 dimension tensor
x.reshape(9, 1) # 9 seperate elements in 2d list

z = x.view(1, 9) # makes a second referencable variable, changes in this variable
# will reflect in x

z[ : , 0] = 5

x[0] # will print 5

# to change stack vectors togeather 

x_stacked = torch.stack([x,x,x,x], dim = 1) # this can be altered
x_stacked


# squeeze remove all single dimensions
non_squeezed_tensor = torch.rand(2,1,3,2,1,2)

squeezed_tensor = torch.squeeze(non_squeezed_tensor)

squeezed_tensor.shape # torch.Size([2, 3, 2, 2])

# unsqueze add a single dimension

a = torch.arange(1,10)
a.shape # torch.Size([9]))

a_unsqueezed = a.unsqueeze(1)
a_unsqueezed, a_unsqueezed.shape

'''(tensor([[1],
         [2],
         [3],
         [4],
         [5],
         [6],
         [7],
         [8],
         [9]]),
 torch.Size([9, 1])) '''
```

changing tensor dimensions
```python
# torch.premute - rearranges dimension of a tensor
a = torch.rand(224, 224, 3) # [height, width, color]

a_redim = torch.permute(a, (2, 0, 1)) # [color, height, width]
```

transform data from numpy to tensor

```python
array = np.arange(1., 8.)

tensor = torch.from_numpy(array) 
# by default numpy create 64 bit int and float
# but tensor use 32 bit so we convert!
tensor.to(torch.float32)

tensor = torch.ones(7)
numpy_arr = tensor.numpy() # note some problem as described above
```
To set a seed for reproducibility
```python
torch.manual_seed(42)
```

To run your code on GPU

```python
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tensor = torch.arange(1.,8.)

# tensor on GPU

tensor_on_gpu = tensor.to(device)

# tensor on gpu cannot be converted to numpy, we need to first transfer the thing onto a cpu
NP = tensor_on_gpu.cpu().numpy()
```

