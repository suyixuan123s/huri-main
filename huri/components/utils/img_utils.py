import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def scale_img(img: np.ndarray, scale: float = 1.):
    """
    Scale image by a factor
    :param img:
    :param scale: scale factor
    :return:
    """
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


def crop_center(img, cropx, cropy, offsetx=0, offsety=0):
    """
    Crop image from center
    :param img:
    :param cropx: image x coordinate
    :param cropy: image y coordinate
    :return: cropped image, (start crop x pos,start crop y pos)
    """
    y, x = img.shape[:2]
    assert y >= cropy and x >= cropx
    startx = max(x // 2 - (cropx // 2) - offsetx, 0)
    starty = max(y // 2 - (cropy // 2) - offsety, 0)
    return img[starty:starty + cropy, startx:startx + cropx], np.array([startx, starty])


def draw_grid(image: np.ndarray, line_space=(8, 8)) -> np.ndarray:
    """
    Draw grid for an image
    :param image:
    :param line_space:
    :return:
    """
    if not isinstance(line_space, list):
        line_space = [line_space, line_space]
    image = image.copy()
    H, W = image.shape[:2]
    image[0:H:line_space[0]] = 1
    image[:, 0:W:line_space[1]] = 1
    return image


def combine_images(images: list[np.ndarray], columns: int = 6, fill: np.uint8 = 255, toggle_sep_dash_line=True):
    assert columns > 0 and len(images) > 0
    # Get image dimensions
    shape_array = np.array([img.shape for img in images])
    image_height, image_width, = shape_array[:, 0].max(), shape_array[:, 1].max()

    _columns = columns
    columns = min(len(images), columns)
    rows = 1 + int(len(images) / _columns)

    # Create a blank image with the same dimensions as the input images
    grid_image = np.zeros((image_height * rows, image_width * columns, 3), dtype=np.uint8)
    grid_image.fill(fill)

    for i, img in enumerate(images):
        y = int(i / _columns)
        x = i % columns
        grid_image[y * image_height:(y + 1) * img.shape[0], x * image_width:(x + 1) * img.shape[1]] = img
        if toggle_sep_dash_line:
            # Add dashed line
            if x < columns - 1:
                x1, y1, x2, y2 = x * image_width + image_width - 1, y * image_height, x * image_width + image_width - 1, (
                        y + 1) * image_height - 1
                for i in range(5, image_height, 10):
                    grid_image[y1 + i, x1] = (0, 0, 0)
                    grid_image[y2 - i, x2] = (0, 0, 0)

            if y < rows - 1:
                x1, y1, x2, y2 = x * image_width, y * image_height + image_height - 1, (
                        x + 1) * image_width - 1, y * image_height + image_height - 1
                for i in range(5, image_width, 10):
                    grid_image[y1, x1 + i] = (0, 0, 0)
                    grid_image[y2, x2 - i] = (0, 0, 0)

    return grid_image


def combine_images2(image_list: list[np.ndarray], columns: int = 6, fill: np.uint8 = 255):
    # Calculate the number of rows required based on the number of columns
    rows = (len(image_list) + columns - 1) // columns

    # Determine the shape of the individual images
    image_shapes = [image.shape for image in image_list]

    # Find the maximum dimensions among the images
    max_height = max(image_shapes, key=lambda x: x[0])[0]
    max_width = max(image_shapes, key=lambda x: x[1])[1]
    num_channels = max(image_shapes, key=lambda x: x[2])[2]

    # Create an empty canvas to hold the combined images
    canvas = np.full((max_height * rows, max_width * columns, num_channels), fill, dtype=np.uint8)

    # Place each image on the canvas
    for idx, image in enumerate(image_list):
        row_idx = idx // columns
        col_idx = idx % columns

        row_start = row_idx * max_height
        row_end = row_start + image.shape[0]
        col_start = col_idx * max_width
        col_end = col_start + image.shape[1]

        canvas[row_start:row_end, col_start:col_end] = image

    return canvas


if __name__ == "__main__":
    # img = cv2.imread("D:\chen\huri_shared\drivers\\rpc\extcam\\1.jpg")
    # new_img = crop_center(img, cropx=2000, cropy=1544)
    # print(new_img.shape)
    # cv2.imshow("fdsfsd", new_img)
    # cv2.waitKey(0)
    # img = cv2.imread("D:\chen\yolo_huri\paper\dataset9\images\\20220310-171004.jpg")
    # img = letterbox(img, new_shape=(1376, 1376), auto=False)[0]
    # cv2.imwrite("stride_8.jpg", draw_grid(img, line_space=8))
    # cv2.imwrite("stride_16.jpg", draw_grid(img, line_space=16))
    # cv2.imwrite("stride_32.jpg", draw_grid(img, line_space=32))
    #
    im1 = crop_center(cv2.imread("D:\chen\yolo_huri\paper\\tase_paper\color_bg\\bg.jpg"), 2064, 1544)[0]
    cv2.imwrite("im1.jpg", im1)
    im2 = crop_center(cv2.imread("D:\chen\yolo_huri\paper\\tase_paper\color_bg\\bg2.jpg"), 2064, 1544)[0]
    cv2.imwrite("im2.jpg", im2)
