import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.measure import block_reduce
from numba import jit
import share_creator

def block_reduce_or(img: np.ndarray, block_size: tuple):
    result = np.bitwise_or.reduceat(np.bitwise_or.reduceat(img, np.arange(0, img.shape[0], block_size), axis=0),
                                      np.arange(0, img.shape[1], block_size), axis=1, dtype=np.uint16)
    return result

def denoise_image(img: np.ndarray):
    denoised = block_reduce_or(img,2)
    return denoised

def __combine_shares__(share1: np.ndarray, share2: np.ndarray):
    combined_share = np.bitwise_and(share1, share2)
    print("combined_share", combined_share[0:10,0:10])
    combined_share = denoise_image(combined_share)
    print("Denoised share", combined_share[0:10,0:10])
    combined_share = gray_to_color(combined_share)
    print("color share", combined_share[0:10,0:10])
    return combined_share

def gray_to_color(img: np.ndarray):
    color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)

    np.bitwise_and(img, 0b1111100000000000, out=color_img[:,:,0])
    np.bitwise_and(img, 0b0000011111000000, out=color_img[:,:,1])
    np.bitwise_and(img, 0b0000000000111110, out=color_img[:,:,2])

    np.right_shift(color_img[:,:,0], 11, out=color_img[:,:,0])
    np.right_shift(color_img[:,:,1], 6 , out=color_img[:,:,1])
    np.right_shift(color_img[:,:,2], 1 , out=color_img[:,:,2])

    color_img = color_img.astype(np.uint8) * 8
    return color_img

def combine_shares(share1: np.ndarray, share2: np.ndarray, verbose=True):
    combined_share = __combine_shares__(share1, share2)
    if verbose:
        plt.imshow(combined_share)
        plt.show()
    return combined_share

if __name__ == "__main__":
    IMAGE = io.imread('images.jpeg')
    share1, share2 = share_creator.generate_shares(IMAGE, verbose=False)
    combine_shares(share1, share2, verbose=True)