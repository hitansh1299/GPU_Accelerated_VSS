import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
from numba import jit, vectorize
import numba as nb
from numba import prange
from scipy.fftpack import dct, idct

BITPLANE_COUNT = 8

'''
Using arithmetic instead of bitwise operations because python has a "fast track" for arithmetic operations
'''
@jit(nopython=True, cache=True, parallel=True)
def rgb_to_ycbcr(img: np.ndarray):
    ycbcr = np.zeros(img.shape, dtype=np.float64)
    img = img.astype(np.float64)

    ycbcr[:,:,0] = img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114 #Y
    ycbcr[:,:,1] = (img[:,:,2] - ycbcr[:,:,0]) * 0.564 + 128 #Cb
    ycbcr[:,:,2] = (img[:,:,0] - ycbcr[:,:,0]) * 0.713 + 128 #Cr
    # print(ycbcr)
    return ycbcr.astype(np.uint8)

'''
    Compresses a 3 channel image into a 2 channel image
    RGB -> Gray Scale, in the ratio 2:3:3, as per this paper: https://www.researchgate.net/publication/269300186_Legibility_of_Web_Page_on_Full_High_Definition_Display
    More complex compression algorithms can be used here, but this is the simplest one, focus in on VSS not compression!
'''
@jit(parallel=True, nopython=True, cache=True)
def compress(img: np.ndarray):
    img = img.astype(np.uint8)
    compressed_img = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    compressed_img = (img[:,:,0] & 0b11000000) | ((img[:,:,1] & 0b111000000) >> 2) | ((img[:,:,2] & 0b11100000) >> 5)
    return compressed_img

@vectorize(nopython=True, cache=True)
def get_bitplane(img: np.ndarray, n: int):
    return (img // 2**n) & 1

@jit(nopython=True, parallel=True, cache=True)
def create_bitplane_shares(img: np.ndarray):
    shares = {
        0: (np.array([[0,1],[0,1]]), np.array([[1,0],[1,0]])),
        1: (np.array([[1,0],[1,0]]), np.array([[0,1],[0,1]])),
        2: (np.array([[1,0],[1,0]]), np.array([[1,0],[1,0]])),
        3: (np.array([[0,1],[0,1]]), np.array([[0,1],[0,1]]))
    }   
    img = img*2 + np.random.randint(0, 2, size=img.shape)
    share1 = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
    share2 = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            share1[i*2:i*2+2, j*2:j*2+2] = shares[img[i][j]][0]
            share2[i*2:i*2+2, j*2:j*2+2] = shares[img[i][j]][1]
    return share1.astype(np.uint8), share2.astype(np.uint8)

#Broken the function into two parts to make to keep the pure numba functions separate from Matplotlib utils
@jit(nopython=True)
def __generate_shares__(img: np.ndarray):
    # img = rgb_to_ycbcr(img)
    img = compress(img)
    share_image1 = np.zeros(shape=(img.shape[0]*2, img.shape[1]*2), dtype=np.uint8)
    share_image2 = np.zeros(shape=(img.shape[0]*2, img.shape[1]*2), dtype=np.uint8)

    for i in range(BITPLANE_COUNT):
        bitplane = get_bitplane(img, i)
        s1, s2 = create_bitplane_shares(bitplane)
        share_image1 = share_image1 + (s1 * 2**i).astype(np.uint8)
        share_image2 = share_image2 + (s2 * 2**i).astype(np.uint8)
    return share_image1, share_image2

def generate_shares(img: np.ndarray, verbose = False):
    share1, share2 = __generate_shares__(img)
    if verbose:
        import share_combiner
        # img = color_to_gray(img)
        plt.figure(figsize=(30,160))
        for i in range(BITPLANE_COUNT):
            plt.subplot(3, BITPLANE_COUNT, i+1)
            plt.axis('off')
            plt.xlabel(f'Bitplane {i}')
            plt.imshow(get_bitplane(img, i), cmap='gray')
            plt.subplot(3, BITPLANE_COUNT, i+BITPLANE_COUNT+1)
            plt.axis('off')
            plt.xlabel(f'Bitplane {i}')
            plt.imshow(get_bitplane(share1,i), cmap='gray')
            plt.subplot(3, BITPLANE_COUNT, i+BITPLANE_COUNT*2+1)
            plt.axis('off')
            plt.xlabel(f'Bitplane {i}')
            plt.imshow(get_bitplane(share2,i), cmap='gray')
        plt.show()
        plt.subplot(1,4,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1,4,2)
        plt.imshow(share1, cmap='gray')
        plt.subplot(1,4,3)
        plt.imshow(share2, cmap='gray')
        plt.subplot(1,4,4)
        combined_share = share_combiner.denoise_image(share1 & share2)
        plt.imshow(combined_share, cmap='gray')
        plt.show()
    return share1, share2


if __name__ == '__main__':
    IMAGE = io.imread('images.jpeg')
    # generate_shares(IMAGE, verbose=True)
    
