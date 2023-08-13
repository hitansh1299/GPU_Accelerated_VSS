import cv2
def split_gif(path: str):
    from PIL import Image
    with Image.open(path) as im:
        num_key_frames =  im.n_frames
        for i in range(num_key_frames):
            im.seek(im.n_frames // num_key_frames * i)
            im.save('frames/{}.png'.format(i))

def split_mp4(path: str, out_path: str='frames/video_frames'):
    try:
        import os
        os.mkdir('frames/video_frames')
    except:
        pass
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(out_path + ("/%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
def convert_to_shares(path: str):
    vidcap = cv2.VideoCapture(path)
    vidwrite = cv2.VideoWriter('share1.avi', fps=0)
    _, frame = vidcap.read()


def split_frames(path: str):
    if path.endswith('.gif'):
        split_gif(path)
    if path.endswith('.mp4') or \
        path.endswith('.mpeg4'):
        split_mp4(path)
    pass

if __name__ == '__main__':
    split_frames('Sample.mp4')