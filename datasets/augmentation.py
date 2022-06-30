import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torch
import cv2


# used only in demo
def crop_contours(path, demo=False):
    cropped_list = []
    img = cv2.imread(path)
    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=40)
    # bin
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((20, 200), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    if demo:
        cv2.imshow('cropped image', dilated)
        cv2.waitKey(0)
    im2, ctrs, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        temp = img[y:y + h, x:x + w]
        # convert cv2 img to pil img
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = Image.fromarray(temp)
        cropped_list.append(np.asarray(temp))
        cv2.rectangle(img, (x, y), (x+w, y+h), (90, 0, 255))
    if demo:
        cv2.imshow('cropped image', img)
        cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return cropped_list, img


def get_image_from_path(img_path, mode, dataset_cfg):
    img = Image.open(img_path).convert('L')
    img = ImageOps.invert(img.point(lambda x: 255 if x > 130 else 0))
    # img.show()
    img = np.asarray(img)
    img = torch.FloatTensor(img)
    img = img[None, :, :]
    img = transforms.Resize(size=(dataset_cfg['dataset']['height'], dataset_cfg['dataset']['width']))(img)
    img /= 255.0

    return img


if __name__ == '__main__':
    crop_contours('../pics/test1.jpg', demo=True)