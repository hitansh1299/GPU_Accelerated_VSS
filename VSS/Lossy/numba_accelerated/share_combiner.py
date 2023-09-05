import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
# from skimage.measure import block_reduce
from numba import jit, prange
from numpy.typing import NDArray

@jit(nopython=True, cache=True, parallel=True)
def __block_reduce_add__(img: np.ndarray, block_size: tuple):
    x = np.zeros((img.shape[0]//block_size[0], img.shape[1]//block_size[1]), dtype=np.uint16)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            x[i][j] = np.sum(img[i:i+block_size[0], j:j+block_size[1]])
            j += block_size[1]
        i += block_size[0]
        
    return x.astype(np.uint8)

@jit(nopython=True, cache=True, parallel=True)
def __block_reduce_or__(img: np.ndarray, block_size: tuple):
    x = np.zeros((img.shape[0]//block_size[0], img.shape[1]//block_size[1]))
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            x[i][j] =   img[i * block_size[0]][j * block_size[1]] | \
                        img[i * block_size[0] + 1][j * block_size[1]] | \
                        img[i * block_size[0]][j * block_size[1] + 1] | \
                        img[i * block_size[0] + 1][j * block_size[1] + 1]
            
    return x.astype(np.uint8)

@jit(nopython=True, cache=True)
def block_reduce_add(img: np.ndarray, block_size: tuple) -> NDArray[np.uint16]:
    # result = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], block_size), axis=0),
    #                                   np.arange(0, img.shape[1], block_size), axis=1, dtype=np.uint16)
    result = __block_reduce_add__(img, block_size)
    return result.astype(np.uint8)

@jit(nopython=True, cache=True)
def block_reduce_or(img: np.ndarray, block_size: tuple) -> NDArray[np.uint16]:
    # result = np.add.reducseat(np.bitwise_or.reduceat(img, np.arange(0, img.shape[0], block_size), axis=0),
    #                                   np.arange(0, img.shape[1], block_size), axis=1, dtype=np.uint16)
    result = __block_reduce_or__(img, block_size)
    return result.astype(np.uint8)

@jit(nopython=True, cache=True)
def denoise_image(img: np.ndarray):
    denoised = block_reduce_or(img, (2,2))
    return denoised

@jit(nopython=True, cache=True)
def __combine_shares__(share1: np.ndarray, share2: np.ndarray):
    
    # combined_share = np.bitwise_and(share1, share2)
    # combined_share = denoise_image(combined_share)
    # combined_share = gray_to_color(combined_share)
    combined_share = np.bitwise_and(share1, share2) 
    combined_share = denoise_image(combined_share)
    combined_share = expand(combined_share)
    return combined_share

'''
    Exapnds the compressed image into a 3 channel image (RGB)
    RGB -> Gray Scale, in the ratio 2:3:3, as per this paper: https://www.researchgate.net/publication/269300186_Legibility_of_Web_Page_on_Full_High_Definition_Display
'''
@jit(parallel=True, nopython=True, cache=True)
def expand(compressed_img: np.ndarray):
    img = np.zeros((compressed_img.shape[0], compressed_img.shape[1], 3))
    img[:,:,0] = (compressed_img & 0b11000000)
    img[:,:,1] = (compressed_img & 0b00111000) << 2 
    img[:,:,2] = (compressed_img & 0b00000111) << 5
    # print('Hello')    
    return img.astype(np.uint8)

# @jit(nopython=True, cache=True)
# def ycbcr_to_rgb(img: np.ndarray):
#     rgb = np.zeros(img.shape, dtype=np.float64)
#     img = img.astype(np.float64)

#     rgb[:,:,0] = img[:,:,0] + (1.403 * (img[:,:,2] - 128))
#     rgb[:,:,1] = img[:,:,0] - (0.343 * (img[:,:,1] - 128)) - (0.714 * (img[:,:,2] - 128))
#     rgb[:,:,2] = img[:,:,0] + (1.773 * (img[:,:,1] - 128))
#     return rgb.astype(np.uint8)

# @jit(nopython=True, cache=True)
# def gray_to_color(img: np.ndarray):
#     color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)

#     color_img[:,:,0] = np.bitwise_and(img, 0b1111100000000000)
#     color_img[:,:,1] = np.bitwise_and(img, 0b0000011111000000)
#     color_img[:,:,2] = np.bitwise_and(img, 0b0000000000111110)

#     color_img[:,:,0] = np.right_shift(color_img[:,:,0], 11)
#     color_img[:,:,1] = np.right_shift(color_img[:,:,1], 6 )
#     color_img[:,:,2] = np.right_shift(color_img[:,:,2], 1 )

#     color_img = color_img.astype(np.uint8) * 8
#     return color_img


def combine_shares(share1: np.ndarray, share2: np.ndarray, verbose=True):
    combined_share = __combine_shares__(share1, share2)
    if verbose:
        plt.imshow(combined_share)
        plt.show()
    return combined_share