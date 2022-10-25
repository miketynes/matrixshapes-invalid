### Automated Matrixshapes Task Generation ###

import numpy as np
import random


INVALIDATABLE = ['multiply', 'add', 'subtract', 'hadamard']

def random_a_shape(dims_max=4):
  """
  Generates random shape for matrix a
  args:
    dims_max: Max amount of dimensions
  returns:
    shape tuple
  """
  dims = np.random.randint(2, dims_max+1, 1)
  shape_a = np.random.randint(2, dims_max, dims)
  return tuple(shape_a)

def invalidate_shape(valid_shape, dim_max):
  """
  given a valid matrix shape, change one of the dimsizes to make it invalid
  """
  new_shape = list(valid_shape)
  dim_invalidate = -2  # for now, always invalidate this one (invalidates np.matmul, and all ops where dims must match)
  valid = True
  while valid:
    new_value = random.randint(1, dim_max)
    valid = new_value == valid_shape[dim_invalidate]
  new_shape[dim_invalidate] = new_value
  return tuple(new_shape)

def multiply_b(shape_a, dim_max=5):
  """
  Generates compatible matrix b to muliply given shape of a 
  args:
    shape_a: shape of matrix a
    dim_max: max value for a given dimension (note this is not the same as dims_max!)
  returns:
    matrix b
  """
  last_axis = np.random.randint(1, dim_max, 1)[0]

  shape_b = list(shape_a)

  shape_b[-2] = shape_a[-1]
  shape_b[-1] = last_axis

  return np.random.randint(0, 5, shape_b)

def same_b(shape_a):
  """
  Generates compatible matrix b for e.g. hadamard product given shape of a 
  """
  return np.random.randint(0, 5, shape_a)

def kron_b(shape_a, dim_max=5):
  """
  Generates compatible matrix b for kronecker product given shape of a 
  """
  shape_b = np.random.randint(2, dim_max, len(shape_a))
  return np.random.randint(0, 5, shape_b)

def sum_axis_b(shape_a):
  """
  Generates a random axis to sum over
  """
  return np.random.randint(0, len(shape_a))

num_to_axis = {
    0: "first",
    1: "second",
    2: "third",
    3: "fourth",
    4: "fifth",
    5: "sixth",
    6: "seventh",
    7: "eighth",
    8: "ninth",
    9: "tenth",
    10: "eleventh"
}

ops = {
    "transpose": {"func": np.transpose, "gen_b": None, "string_beg": "Transpose a matrix of shape {}.", "string_mid": "Transpose the result."},
    "multiply": {"func": np.matmul, "gen_b": multiply_b, "string_beg": "Multiply a matrix of shape {} with a matrix of shape {}.", "string_mid": "Multiply the result with a matrix of shape {}."},
    "hadamard": {"func": np.multiply, "gen_b": same_b, "string_beg": "Compute the hadamard product of a matrix of shape {} with a matrix of shape {}.", "string_mid": "Compute the hadamard product of the result with a matrix of shape {}."},
    "add": {"func": np.add, "gen_b": same_b, "string_beg": "Add a matrix of shape {} to a matrix of shape {}.", "string_mid": "Add the result to a matrix of shape {}."},
    "subtract": {"func": np.subtract, "gen_b": same_b, "string_beg": "Subtract a matrix of shape {} from a matrix of shape {}.", "string_mid": "Subtract the result from a matrix of shape {}."},
    "kronecker": {"func": np.kron, "gen_b": kron_b, "string_beg": "Compute the kronecker product of a matrix of shape {} with a matrix of shape {}.", "string_mid": "Compute the kronecker product of the result with a matrix of shape {}."},
    "sum_axis": {"func": np.sum, "gen_b": sum_axis_b, "string_beg": "Take a matrix of shape {} and sum over the {} axis.", "string_mid": "Sum the result over the {} axis."}
}

def generate(shape_start=None, num_ops=5, dim_max=5, dims_max=4, invalid=False):
    """
    Generates an input example for the matrixshapes language task
    args: 
      shape_start: Tuple of starter shape, e.g. (1, 2)
      num_ops: int of how many operations to apply at max
      dim_max: Maximum value of a dimension
      dims_max: Maximum number of dimensions
    returns:
      input: string representing the model input
      label: string repreenting the model label
      boolean value if the label appears in the input
    """
    input = ""
    if shape_start is None:
      shape_start = random_a_shape(dims_max=dims_max)
    num_ops = random.randint(1, num_ops)  # there should always be at leaset 1 operation, and counts should start at 1!
    op_invalidate = random.randint(0, num_ops-1) if invalid else -1  # but indexing starts at zero.
    this_op_invalid = op_invalidate == 0

    # Keep track of shapes that are print out for confounders
    shapes = [shape_start]

    a = np.random.randint(0, dim_max, shape_start)

    # Generate first operation & select its subdict
    op_beg_str, op_beg = random.choice(list(ops.items()))

    # Summing over axis requires at least 3 axes
    while ((op_beg["gen_b"] == sum_axis_b) and (len(a.shape) < 3)) \
            or (this_op_invalid and (op_beg_str not in INVALIDATABLE)) :
      op_beg_str, op_beg = random.choice(list(ops.items()))

    if op_beg["gen_b"] is None:
      a = op_beg["func"](a)
      input += op_beg["string_beg"].format(str(shape_start).replace(" ", ""))
    else:
      b = op_beg["gen_b"](shape_start)
      
      # Add to input string
      if op_beg["gen_b"] == sum_axis_b:
        input += op_beg["string_beg"].format(str(shape_start).replace(" ", ""), num_to_axis[b])
      else:
        shape = invalidate_shape(b.shape, dim_max=dim_max) if this_op_invalid else b.shape
        input += op_beg["string_beg"].format(str(shape_start).replace(" ", ""), str(shape).replace(" ", ""))
        shapes.append(b.shape)

      # Keep track of solution shape
      a = op_beg["func"](a, b)

    # Generate random list of operations after the zero'th operation
    for i in range(1, num_ops):  # this will be empty if num_ops is 1, because it starts at 1, but range is exclusive!
      op_mid_str, op_mid = random.choice(list(ops.items()))
      this_op_invalid = i == op_invalidate
      # Summing over axis requires at least 3 axes
      while ((op_mid["gen_b"] == sum_axis_b) and (len(a.shape) < 3)) \
            or (this_op_invalid and not (op_mid_str in INVALIDATABLE)):
       op_mid_str, op_mid = random.choice(list(ops.items()))

      input += " "

      if op_mid["gen_b"] is None:
        a = op_mid["func"](a)
        input += op_mid["string_mid"]
      else:
        b = op_mid["gen_b"](a.shape)

        # Add to input string
        if op_mid["gen_b"] == sum_axis_b:
          input += op_mid["string_mid"].format(num_to_axis[b])
        else:
          shape = b.shape if not this_op_invalid else invalidate_shape(b.shape, dim_max)
          input += op_mid["string_mid"].format(str(shape).replace(" ", ""))
          shapes.append(b.shape)

        # Keep track of solution shape
        a = op_mid["func"](a, b)

    output_shape = "Invalid." if invalid else str(a.shape).replace(" ", "")
    return input, output_shape, a.shape in shapes


if __name__ == '__main__':
  
    # Notes:
    # If you use too many operations it may well crash due to the kronecker product; The default is 1-5
    out = generate()
    
    print(out)
