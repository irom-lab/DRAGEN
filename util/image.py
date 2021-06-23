from PIL import Image, ImageFilter
import numpy as np


def process_image(image_arr, blur_strength=3):
    image = Image.fromarray(np.moveaxis(np.uint8(image_arr*255), 0, -1))
        # from CHW to HWC
    if blur_strength > 0:
        processed = image.filter(ImageFilter.GaussianBlur(blur_strength))
    else:
        processed = image
    processed_arr = np.array(processed)
    return processed_arr    # HWC
