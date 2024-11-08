from PIL import Image
import cv2

from huri.core.common_import import fs


def conv2eps(file_path):
    file_path = fs.Path(file_path).resolve()
    img_paths = []
    if file_path.is_dir():
        for f in file_path.glob("*"):
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_paths.append(f)
    elif file_path.is_file():
        if file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_paths.append(file_path)
    for img_path in img_paths:
        img_ppath = img_path.parent
        img = Image.open(str(img_path))
        img.save(str(img_ppath.joinpath(f"{img_path.name.split('.')[0]}.eps")), lossless=True)


def remove_white_space(file_path):
    file_path = fs.Path(file_path).resolve()
    img_paths = []
    if file_path.is_dir():
        for f in file_path.glob("*"):
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_paths.append(f)
    elif file_path.is_file():
        if file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_paths.append(file_path)
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img_ppath = img_path.parent
        img_name = img_path.name.split(".")[0]

        # process
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        # get contours
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) < 1:
            return
        x = 999999999
        y = 999999999
        x2 = 0
        y2 = 0
        for cnt in cnts:
            ## (4) Crop and save it
            x_1, y_1, w, h = cv2.boundingRect(cnt)
            x_2 = x_1 + w
            y_2 = y_1 + h
            x = min(x, x_1)
            y = min(y, y_1)
            x2 = max(x2, x_2)
            y2 = max(y2, y_2)
        print(cv2.boundingRect(sorted(cnts, key=cv2.contourArea)[-1]))
        print(x, y, x2, y2)
        dst = img[y:y2, x:x2]

        cv2.imwrite(str(img_ppath.joinpath(f"{img_name}.jpg")), dst)

if __name__ == "__main__":
    # remove_white_space("C:\\Users\WRS\Desktop\\tmp\\rsj2022")
    remove_white_space("C:\\Users\\WRS\\Desktop\\data\\RAL2022")

    # conv2eps("C:\\Users\WRS\Desktop\\tmp\\rsj2022")
    # remove_white_space(f"robot_movement.jpg")