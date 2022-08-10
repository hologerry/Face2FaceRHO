import os

import cv2
import numpy as np


# img_path = "val_case/000136.png"

# img_path_512 = "val_case/000136_512.png"

# img_path_256 = "val_case/000136_256.png"

# img = cv2.imread(img_path)
# img_512 = cv2.resize(img, (512, 512))
# img_256 = cv2.resize(img, (256, 256))

# cv2.imwrite(img_path_512, img_512)
# cv2.imwrite(img_path_256, img_256)


def crop_square(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


video_dir = "val_case_video"
video_dir_512 = "val_case_video_512"
os.makedirs(video_dir_512, exist_ok=True)

img_files = os.listdir(video_dir)
for img_file in img_files:
    img_path = os.path.join(video_dir, img_file)
    img_path_512 = os.path.join(video_dir_512, img_file)
    img = cv2.imread(img_path)
    img_512 = crop_square(img, 512)
    cv2.imwrite(img_path_512, img_512)
