# matrixshapes
This task primes language models to keep track of matrixshapes, with the possibility of invalid matrix operations. 
This is an extension of Niklas Muenninghoff's [matrixshapes]("https://github.com/Muennighoff/matrixshapes") work. 
The goal is to see if transformers can detect whether a mathematical operation is correctly specified. 

Requires: `numpy`

To generate a dataset, run: `git clone https://github.com/miketynes/matrixshapes.git`
and cd into the repository.
Then run `python generate.py`. Arguments are identical to those in Niklas Muenninghoff's 
[matrixshapes]("https://github.com/Muennighoff/matrixshapes") work, with one additional argument: 
* `--frac_invalid, float, default=0.5`, the fraction of examples where the given chain of matrix operations is invalid
and one changed default: 
* `--num` (number of examples), defaults to 100 instead of 1000.

**Supported operations**:   
All matrix operations are on N-D arrays with N $\geq 2$. Operations follow `numpy` conventions.
* Addition
* Subtraction
* Hadamard (elementwise) multiplication
* Summing over a dimension
* Transpose (reverse order of axes)
* Matrix Multiplcation (numpy style: $AB$ where $A$ and $B$ are order $n$ tensors and the last index of $A$ 
  must have the same size as the second-to-last index of $B$). 
  Quoting the [numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) 
  "If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly."
* Kronecker Product. Quoting the [numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.kron.html)
  "If `a.shape = (r0,r1,..,rN)` and `b.shape = (s0,s1,...,sN)`, the Kronecker product has shape `(r0*s0, r1*s1, ..., rN*SN)`"

**Possibly invalid operations**
Operations on two arrays which have constraints on the input matrix shapes may be invalidated. 
This includes: Addition, Subtraction, Hadamard multiplication, and matrix multiplication.
At most one operation will be invalidated per example.

**Possible improvements**
* Invalidate summing operations (e.g., sum over an axis that is not present)
* Invalidate Kronecker Product (mismacth in N-D)
* Invalidate more than one operation per example
* More flexible selection of index of invalidation
