import numpy as np
import cv2
from VSS.Lossless.numba_accelerated import share_creator
def create_shares(input: str, out1='share1.avi', out2='share2.avi'):
    cap = cv2.VideoCapture(0)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'HFYU')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    share1 = cv2.VideoWriter('share1.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2),
                                                                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)))
    share2 = cv2.VideoWriter('share2.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2),
                                                                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            s1, s2 = share_creator.generate_shares(frame)
            # print(frame.shape)
            # write the flipped frame
            share1.write(s1, )
            share2.write(s2)
            cv2.imshow('Video',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    share1.release()
    cv2.destroyAllWindows()