import matplotlib.pyplot as plt
import skimage.io as io
from skimage.measure import block_reduce
import numpy as np
from numba import jit, vectorize
import numba as nb


IMAGE = io.imread('images.jpeg')
BITPLANE_COUNT = 16
# plt.imshow(IMAGE)

'''
Using arithmetic instead of bitwise operations because python has a "fast track" for arithmetic operations
'''
@jit(nopython=True)
def color_to_gray(img: np.ndarray):
    gray_img = np.zeros(img.shape, dtype=np.uint16)
    get_first_5_bits = lambda x: x // 0b1000
    gray_img = get_first_5_bits(img[:, :, 0]) * 2048 + get_first_5_bits(img[:, :, 1]) * 64 + get_first_5_bits(img[:, :, 2]) * 2
    # plt.imshow(gray_img, cmap='gray')
    return gray_img

@vectorize(nopython=True)
def get_bitplane(img: np.ndarray, n: int):
    return (img // 2**n) & 1

@jit(nopython=True)
def create_bitplane_shares(img: np.ndarray):
    shares = {
        3: (np.array([[0,1],[0,1]]), np.array([[1,0],[1,0]])),
        2: (np.array([[1,0],[1,0]]), np.array([[0,1],[0,1]])),
        1: (np.array([[1,0],[1,0]]), np.array([[1,0],[1,0]])),
        0: (np.array([[0,1],[0,1]]), np.array([[0,1],[0,1]]))
    }   
    img = img*2 + np.random.randint(0, 2, size=img.shape)
    share1 = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
    share2 = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            share1[i*2:i*2+2, j*2:j*2+2] = shares[img[i][j]][0]
            share2[i*2:i*2+2, j*2:j*2+2] = shares[img[i][j]][1]
    return share1.astype(np.uint16), share2.astype(np.uint16)

def denoise_rebuilt_image(img: np.ndarray):
    denoised = block_reduce(img, (2, 2), np.sum) #TODO: Replace this with a vectorizable algorithm
    denoised = np.kron(denoised, np.ones((2, 2)))
    # plt.imshow(denoised, cmap='gray')
    return denoised

#Broken the function into two parts to make to keep the pure numba functions separate from Matplotlib utils
@jit(nopython=True)
def __generate_shares__(img: np.ndarray):
    img = color_to_gray(img)
    share_image1 = np.zeros(shape=(img.shape[0]*2, img.shape[1]*2), dtype=np.uint16)
    share_image2 = np.zeros(shape=(img.shape[0]*2, img.shape[1]*2), dtype=np.uint16)
    # print(img.shape)
    for i in range(BITPLANE_COUNT):
        bitplane = get_bitplane(img, i)
        # print(i, bitplane.shape)
        s1, s2 = create_bitplane_shares(bitplane)
        # print(s1.shape, s2.shape)
        share_image1 = share_image1 + (s1 * 2**i).astype(np.uint16)
        share_image2 = share_image2 + (s2 * 2**i).astype(np.uint16)
        # print('All done here')
    return share_image1, share_image2

def generate_shares(img: np.ndarray, verbose = False):
    share1, share2 = __generate_shares__(img)
    print(share1)
    if verbose:
        img = color_to_gray(img)
        plt.figure(figsize=(30,160))
        for i in range(BITPLANE_COUNT):
            plt.subplot(3, BITPLANE_COUNT, i+1)
            plt.axis('off')
            plt.imshow(get_bitplane(img, i), cmap='gray')
            plt.subplot(3, BITPLANE_COUNT, i+BITPLANE_COUNT+1)
            plt.axis('off')
            plt.imshow(get_bitplane(share1,i), cmap='gray')
            plt.subplot(3, BITPLANE_COUNT, i+BITPLANE_COUNT*2+1)
            plt.axis('off')
            plt.imshow(get_bitplane(share2,i), cmap='gray')
        plt.show()
        plt.subplot(1,4,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1,4,2)
        plt.imshow(share1, cmap='gray')
        plt.subplot(1,4,3)
        plt.imshow(share2, cmap='gray')
        plt.subplot(1,4,4)
        plt.imshow(denoise_rebuilt_image(share1 | share2), cmap='gray')
        plt.show()
    return share1, share2

import timeit
if __name__ == '__main__':
    generate_shares(IMAGE, verbose=False)