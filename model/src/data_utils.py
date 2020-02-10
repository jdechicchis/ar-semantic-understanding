"""
Image and data manipulation utility functions.
"""

import math
import numpy as np
from PIL import Image

def horizontal_flip(image, mask):
    """
    Apply horizontal flip.
    """
    assert image.size == mask.size, "Image and mask dimensions must match"
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask

def rotate(image, mask, angle):
    """
    Rotate by some angle in degrees.
    """
    assert image.size == mask.size, "Image and mask dimensions must match"
    original_width = image.size[0]
    original_height = image.size[1]
    image = image.rotate(angle, Image.BILINEAR)
    mask = mask.rotate(angle, Image.NEAREST)
    new_width, new_height = largest_rotated_rect(original_width, original_height, angle)
    image = center_crop(image, new_width, new_height)
    mask = center_crop(mask, new_width, new_height)
    image = image.resize((original_width, original_height), Image.BILINEAR)
    mask = mask.resize((original_width, original_height), Image.NEAREST)
    return image, mask

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size w x h that has been rotated by 'angle' (in
    degrees), compute the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    From:
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    angle = math.radians(angle)

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        int(bb_w - 2 * x),
        int(bb_h - 2 * y)
    )

def center_crop(image, width, height):
    """
    Center crop image to width x height.
    """
    top_left_x = (image.size[0] - width) / 2
    top_left_y = (image.size[1] - height) / 2
    bottom_right_x = image.size[0] - (image.size[0] - width) / 2
    bottom_right_y = image.size[1] - (image.size[1] - height) / 2
    image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    return image

def horizontal_shear(image, mask, amount):
    """
    Horizontal shear by some amount (float).
    """
    assert image.size == mask.size, "Image and mask dimensions must match"
    width = image.size[0]
    height = image.size[1]

    mask = np.array(mask)

    image_mask = np.zeros((width, height, 3), dtype=np.uint8)
    for w in range(0, 224):
        for h in range(0, 224):
            image_mask[h][w] = [mask[h][w]]

    mask = Image.fromarray(image_mask)

    new_image = Image.new("RGB", (width * 2, height), (0, 0, 0))
    new_mask = Image.new("RGB", (width * 2, height), (0, 0, 0))
    if amount >= 0:
        new_image.paste(image, (width, 0))
        new_mask.paste(mask, (width, 0))
    else:
        new_image.paste(image, (0, 0))
        new_mask.paste(mask, (0, 0))
    image = new_image
    mask = new_mask

    image = image.transform(image.size, Image.AFFINE, (1, amount, 0, 0, 1, 0))
    mask = mask.transform(image.size, Image.AFFINE, (1, amount, 0, 0, 1, 0))

    if amount >= 0:
        image = image.crop((width*(1-amount), 0, image.size[0], height))
        image = image.crop((width*amount, 0, image.size[0] - width*amount, height))
        mask = mask.crop((width*(1-amount), 0, mask.size[0], height))
        mask = mask.crop((width*amount, 0, mask.size[0] - width*amount, height))
    else:
        image = image.crop((0, 0, image.size[0] - width*(1-abs(amount)), height))
        image = image.crop((width*abs(amount), 0, image.size[0] - width*abs(amount), height))
        mask = mask.crop((0, 0, mask.size[0] - width*(1-abs(amount)), height))
        mask = mask.crop((width*abs(amount), 0, mask.size[0] - width*abs(amount), height))

    assert image.size == mask.size, "Image and mask size must match after shear"

    crop_width_height = min(image.size[0], image.size[1])
    image = center_crop(image, crop_width_height, crop_width_height)
    mask = center_crop(mask, crop_width_height, crop_width_height)

    image = image.resize((width, height), Image.BILINEAR)
    mask = mask.resize((width, height), Image.NEAREST)

    mask = np.array(mask)
    return_mask = np.zeros((width, height), dtype=np.uint8)
    for w in range(0, width):
        for h in range(0, height):
            return_mask[h][w] = mask[h][w][0]

    return image, Image.fromarray(return_mask)
