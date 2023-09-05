import numpy as np
import cv2
from VSS.Lossless.numba_accelerated import share_combiner

def __combine_video__(share1:str = 'share1.avi', share2:str = 'share2.avi', output='output.avi', verbose=True):
    share1 = cv2.VideoCapture('share1.avi')
    share2 = cv2.VideoCapture('share2.avi')
    if (share1.isOpened() == False or share2.isOpened() == False): 
        print("Error opening video stream or file")
    # print(share1.get(cv2.CAP_PROP_FRAME_WIDTH), share1.get(cv2.CAP_PROP_FRAME_HEIGHT), share1.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'HFYU')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    combined = cv2.VideoWriter('output.avi', fourcc, share1.get(cv2.CAP_PROP_FPS), (int(share1.get(cv2.CAP_PROP_FRAME_WIDTH) * 2),
                                                                               int(share1.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)))

    while(share1.isOpened() and share2.isOpened()):
        ret1, frame1 = share1.read()
        ret2, frame2 = share2.read()
        if ret1==True and ret2 == True:
            combined = share_combiner.combine_shares(frame1, frame2, verbose=False)
            # print(frame.shape)
            # write the flipped frame
            cv2.imshow('Video', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    share1.release()
    share1.release()
    cv2.destroyAllWindows()
