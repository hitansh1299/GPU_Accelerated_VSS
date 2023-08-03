from PIL import Image
with Image.open('sample.gif') as im:
    num_key_frames =  im.n_frames
    for i in range(num_key_frames):
        im.seek(im.n_frames // num_key_frames * i)
        im.save('frames/{}.png'.format(i))