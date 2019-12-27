#### Electronic Insert I.1 – DCT-II / DCT-III code generator

```python
#######################################################################
# DCT-II / DCT-III generator
#
# Based on:
#  "A low multiplicative complexity fast recursive DCT-2 algorithm"
#  by Maxim Vashkevich and Alexander Petrovsky / arXiv / 20 Jul 2012
#######################################################################

import math
import sys
N = 8

#######################################################################
# Base transforms / generators
#######################################################################

CNTR = 0
def makeTmp():
  global CNTR
  result = "t{:02d}".format(CNTR)
  CNTR = CNTR + 1
  return result

def makeVar(i):
  return "i{:02d}".format(i)

def add(x, y):
  tmp = makeTmp()
  print(tmp + " = " + x + " + " + y + ";")
  return tmp

def sub(x, y):
  tmp = makeTmp()
  print(tmp + " = " + x + " - " + y + ";")
  return tmp

def mul(x, c):
  tmp = makeTmp()
  print(tmp + " = " + x + " * " + c + ";")
  return tmp

# 2.0 * math.cos((a + 0.0) / (b + 0.0) * math.pi)
def C2(a, b):
  return "c_c2_" + str(a) + "_" + str(b)

# 1.0 / C2(a, b)
def iC2(a, b):
  return "c_ic2_" + str(a) + "_" + str(b)

#######################################################################
# Utilities
#######################################################################

# Generate identity matrix. Usually this matrix is passed to
# DCT algorithm to generate "basis" vectors of the transform.
def makeVars():
  return [makeVar(i) for i in range(N)]

# Split list of variables info halves.
def split(x):
  m = len(x)
  m2 = m // 2
  return (x[0 : m2], x[m2 : m])

# Make a list of variables in a reverse order.
def reverse(varz):
  m = len(varz)
  result = [0] * m
  for i in range(m):
    result[i] = varz[m - 1 - i]
  return result

# Apply permutation
def permute(x, p):
 return [x[p[i]] for i in range(len(p))]

def transposePermutation(p):
  n = len(p)
  result = [0] * n
  for i in range(n):
    result[p[i]] = i
  return result

# See paper. Split even-odd elements.
def P(n):
  if n == 1:
    return [0]
  n2 = n // 2
  return [2 * i for i in range(n2)] + [2 * i + 1 for i in range(n2)]

# See paper. Interleave first and second half.
def Pt(n):
  return transposePermutation(P(n))

#######################################################################
# Scheme
#######################################################################

def B2(x):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  bottom = reverse(bottom)
  t = [add(top[i], bottom[i]) for i in range(n2)]
  b = [sub(top[i], bottom[i]) for i in range(n2)]
  return t + b

def iB2(x):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  t = [add(top[i], bottom[i]) for i in range(n2)]
  b = [sub(top[i], bottom[i]) for i in range(n2)]
  return t + reverse(b)

def B4(x, rn):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  rbottom = reverse(bottom)
  t = [sub(top[i], rbottom[i]) for i in range(n2)]
  b = [mul(bottom[i], C2(rn, 2 * N)) for i in range(n2)]
  top = [add(t[i], b[i]) for i in range(n2)]
  bottom = [sub(t[i], b[i]) for i in range(n2)]
  return top + bottom

def iB4(x, rn):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  t = [add(top[i], bottom[i]) for i in range(n2)]
  b = [sub(top[i], bottom[i]) for i in range(n2)]
  bottom = [mul(b[i], iC2(rn, 2 * N)) for i in range(n2)]
  rbottom = reverse(bottom)
  top = [add(t[i], rbottom[i]) for i in range(n2)]
  return top + bottom

def P4(n):
  if n == 1:
    return [0]
  if n == 2:
    return [0, 1]
  n2 = n // 2
  result = [0] * n
  tc = 0
  bc = 0
  i = 0
  result[i] = tc; tc = tc + 1; i = i + 1
  turn = True
  while i < n - 1:
    if turn:
      result[i] = n2 + bc; bc = bc + 1; i = i + 1
      result[i] = n2 + bc; bc = bc + 1; i = i + 1
    else:
      result[i] = tc; tc = tc + 1; i = i + 1
      result[i] = tc; tc = tc + 1; i = i + 1
    turn = not turn
  result[i] = tc; tc = tc + 1; i = i + 1
  return result

def iP4(n):
  return transposePermutation(P4(n))

def d2n(x):
  n = len(x)
  if n == 1:
    return x
  y = B2(x)
  (top, bottom) = split(y)
  return permute(d2n(top) + d4n(bottom, N // 2), Pt(n))

def id2n(x):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(permute(x, P(n)))
  return iB2(id2n(top) + id4n(bottom, N // 2))

def d4n(x, rn):
  n = len(x)
  if n == 1:
    return x
  y = B4(x, rn)
  (top, bottom) = split(y)
  rn2 = rn // 2
  return permute(d4n(top, rn2) + d4n(bottom, N - rn2), P4(n))

def id4n(x, rn):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(permute(x, iP4(n)))
  rn2 = rn // 2
  y = id4n(top, rn2) + id4n(bottom, N -rn2)
  return iB4(y, rn)

#######################################################################
# Main.
#######################################################################

def help():
  print("Usage: %s [N [T]]" % sys.argv[0])
  print("  N should be the power of 2, default is 8")
  print("  T is one of {2, 3}, default is 2")
  sys.exit()

def parseInt(s):
  try:
    return int(s)
  except ValueError:
    help()

if __name__ == "__main__":
  if len(sys.argv) < 1 or len(sys.argv) > 3: help()
  if len(sys.argv) >= 2:
    N = parseInt(sys.argv[1])
    if (N & (N - 1)) != 0: help()
  type = 0
  if len(sys.argv) >= 3:
    typeOption = sys.argv[2]
    if len(typeOption) != 1: help()
    type = "23".index(typeOption)
    if type == -1: help()
  if type == 0:
    vars = d2n(makeVars())
  else:  # type == 1
    vars = id2n(makeVars())
  print("Output vector: " + str(vars))
```

