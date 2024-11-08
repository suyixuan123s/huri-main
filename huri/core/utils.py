"""
Some utils functions
"""
from huri.core.common_import import np


def color_hex2oct(hex_color, alpha=1):
    """
    Change hex color to oct color
    :param hex_color: hex color
    :param alpha: alpha channel
    :return: RGBA color
    """
    hex_color = hex_color.replace('#', '').replace(" ", '')
    if len(hex_color) == 6:
        return np.array(
            [int(hex_color[:2], 16) / 255, int(hex_color[2:4], 16) / 255, int(hex_color[4:], 16) / 255, alpha])
    else:
        raise Exception("hex color has problem")


# some hex color
color_hex = {
    "Beach Towels": ["#fe4a49", "#2ab7ca", "#fed766", "#e6e6ea", "#f4f4f8"],
    "Light Pink": ['#eee3e7', '#ead5dc', '#eec9d2', '#f4b6c2', '#f6abb6'],
    "Beautiful Blues": ['#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0'],
    "So Many Lost Songs": ['#051e3e', '#251e3e', '#451e3e', '#651e3e', '#851e3e'],
    "She": ['#dec3c3', '#e7d3d3', '#f0e4e4', '#f9f4f4', '#ffffff'],
    "Moonlight Bytes 6": ['#4a4e4d', '#0e9aa7', '#3da4ab', '#f6cd61', '#fe8a71'],
    "Number 3": ['#2a4d69', '#4b86b4', '#adcbe3', '#e7eff6', '#63ace5'],
}

if __name__ == "__main__":
    a = img_to_n_channel(np.ones((3, 3)))
    a[0, 0, 1] = 2
    print(a)
