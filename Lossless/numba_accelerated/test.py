from share_creator import generate_shares
from share_combiner import combine_shares
from skimage import io


if __name__ == '__main__':
    IMAGE = io.imread('images.jpeg')
    share1, share2 = generate_shares(IMAGE, verbose=False)
    share = combine_shares(share1, share2, verbose=False)
