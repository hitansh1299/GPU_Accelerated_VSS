import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
# from skimage.measure import block_reduce
from numba import jit, prange
from numpy.typing import NDArray

PARALLEL = True
@jit(nopython=True, cache=True, parallel=PARALLEL)
def __block_reduce_add__(img: np.ndarray, block_size: tuple):
    x = np.zeros((img.shape[0]//block_size[0], img.shape[1]//block_size[1]), dtype=np.uint16)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            x[i][j] = np.sum(img[i:i+block_size[0], j:j+block_size[1]])
            j += block_size[1]
        i += block_size[0]
        
    return x.astype(np.uint8)

@jit(nopython=True, cache=True, parallel=PARALLEL)
def __block_reduce_or__(img: np.ndarray, block_size: tuple):
    x = np.zeros((img.shape[0]//block_size[0], img.shape[1]//block_size[1]))
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            x[i][j] =   img[i * block_size[0]][j * block_size[1]] | \
                        img[i * block_size[0] + 1][j * block_size[1]] | \
                        img[i * block_size[0]][j * block_size[1] + 1] | \
                        img[i * block_size[0] + 1][j * block_size[1] + 1]            
    return x.astype(np.uint8)

@jit(nopython=True, cache=True, parallel=PARALLEL)
def block_reduce_add(img: np.ndarray, block_size: tuple) -> NDArray[np.uint16]:
    result = __block_reduce_add__(img, block_size)
    return result.astype(np.uint8)

@jit(nopython=True, cache=True)
def block_reduce_or(img: np.ndarray, block_size: tuple) -> NDArray[np.uint16]:
    result = __block_reduce_or__(img, block_size)
    return result.astype(np.uint8)

@jit(nopython=True, cache=True)
def denoise_image(img: np.ndarray):
    denoised = block_reduce_or(img, (2,2))

    return denoised

@jit(nopython=True, cache=True, parallel=PARALLEL)
def __combine_shares__(share1: np.ndarray, share2: np.ndarray):
    combined_share = np.bitwise_and(share1, share2) 
    combined_share = np.dstack((denoise_image(combined_share[:,:,0]),
                               denoise_image(combined_share[:,:,1]),
                               denoise_image(combined_share[:,:,2])))


    # combined_share[:,:,0] = gray_to_color(combined_share)
    return combined_share

def combine_shares(share1: np.ndarray, share2: np.ndarray, verbose=True, high_res=False):
    print('combining shares')
    combined_share = __combine_shares__(share1, share2)
    if verbose:
        plt.imshow(combined_share)
        plt.show()
    return combined_share
