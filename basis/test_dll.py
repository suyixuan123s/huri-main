import ctypes
import numpy as np
import basis.robot_math as rm

## CPP
## https://blog.janjan.net/2021/01/26/vsc-python-import-windows-dll-error-function-not-found/

# find the shared library, the path depends on the platform and Python version
# 1. open the shared library
robot_math_c = ctypes.cdll.LoadLibrary("./robotmath_fast_ctype.dll")

# 2. tell Python the argument and result types of function
robot_math_c.rotmat_from_axangle.restype = np.ctypeslib.ndpointer(dtype=np.double, shape=(3, 3), flags="C_CONTIGUOUS")
robot_math_c.rotmat_from_axangle.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, shape=(1,3), flags="C_CONTIGUOUS"),
                                             ctypes.c_double]
# 3. call function
import timeit
t = timeit.timeit(lambda:robot_math_c.rotmat_from_axangle(np.array([[0, 1, 0]], dtype=np.double), np.pi / 3), number=12000)
print(t)
t = timeit.timeit(lambda:rm.rotmat_from_axangle(np.array([0, 1, 0], dtype=np.double), np.pi / 3), number=12000)
print(t)