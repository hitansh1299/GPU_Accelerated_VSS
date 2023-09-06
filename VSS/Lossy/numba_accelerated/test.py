from share_creator import generate_shares
from share_combiner import combine_shares
from skimage import io
import matplotlib.pyplot as plt
import numba
if __name__ == '__main__':
    IMAGE = io.imread('lena.png')
    s1, s2 = generate_shares(IMAGE, verbose=False)
    combine_shares(s1, s2, verbose=True)