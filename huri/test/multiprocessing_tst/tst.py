import numpy as np
import time

n_a_rows = 4000
n_a_cols = 3000
n_b_rows = n_a_cols
n_b_cols = 200

a = np.arange(n_a_rows * n_a_cols).reshape(n_a_rows, n_a_cols)
b = np.arange(n_b_rows * n_b_cols).reshape(n_b_rows, n_b_cols)

start = time.time()
d = np.dot(a, b)
end = time.time()

print("time taken : {}".format(end - start))
print(a)
print(b)
print(d)