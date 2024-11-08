"""
Beautiful print tool
"""

use_pd = True
try:
    import pandas as pd
except:
    print("Cannot import pandas, Using normal print mode")
    use_pd = False
from numpy import ndarray, arange


def print_array(a, cols, rows):
    """
    Print array in a better format
    :param a:
    :param cols:
    :param rows:
    :return:
    """
    if (len(cols) != a.shape[1]) or (len(rows) != a.shape[0]):
        print("Shapes do not match")
        return
    s = a.__repr__()
    s = s.split("array(")[1]
    s = s.replace("      ", "")
    s = s.replace("[[", " [")
    s = s.replace("]])", "]")
    pos = [i for i, ltr in enumerate(s.splitlines()[0]) if ltr == ","]
    pos[-1] = pos[-1] - 1
    empty = " " * len(s.splitlines()[0])
    s = s.replace("],", "]")
    s = s.replace(",", "")
    lines = []
    for i, l in enumerate(s.splitlines()):
        lines.append(rows[i] + l)
    s = "\n".join(lines)
    empty = list(empty)
    for i, p in enumerate(pos):
        empty[p - i] = cols[i]
    s = "".join(empty) + "\n" + s
    print(s)


def text_pd(mtx: ndarray, dtype=int, precision=None):
    """
    print array in a pandas format
    :param mtx: matrix
    :param dtype: data type for the matrix
    :param precision: if dtype is float, set the precision to show
    :return:
    """
    row, cols = mtx.shape[0], mtx.shape[1]
    if cols is None:
        df = pd.DataFrame(mtx, index=arange(row), dtype=dtype)
    else:
        df = pd.DataFrame(mtx, columns=arange(cols), index=arange(row), dtype=dtype)
    if precision is not None:
        df.round(precision)
    return df


def chunkstring(string, length):
    return (string[0 + i:length + i] for i in range(0, len(string), length))


def print_with_border(text, width=50):
    """
    Print text with a border
    :param text:
    :param width:
    :return:
    """
    print('+-' + '-' * width + '-+')

    for line in chunkstring(text, width):
        print('| {0:^{1}} |'.format(line, width))

    print('+-' + '-' * (width) + '-+')
