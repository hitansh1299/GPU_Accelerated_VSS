import numpy as np
import cv2
from VSS.Lossless.numba_accelerated import share_creator as share_creator_lossless
from VSS.Lossy.numba_accelerated import share_creator as share_creator_lossy

def create_shares(input: str | int, out1='share1.avi', out2='share2.avi', mode='lossless', high_res=True):
    cap = cv2.VideoCapture(input)
    if (cap.isOpened()== False): 
        # print("Error opening video stream or file")
        raise Exception("Error opening video stream or file")
    
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'FFV1')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    if mode == 'lossy':
        if high_res:
            share1 = cv2.VideoWriter(out1, 
                                    fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)), 
                                    params=[
                                        cv2.VIDEOWRITER_PROP_DEPTH,
                                        cv2.CV_16U,
                                        cv2.VIDEOWRITER_PROP_IS_COLOR,
                                        0  # false
                                    ]
                                )
            share2 = cv2.VideoWriter(out2, 
                                    fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)), 
                                    params=[
                                        cv2.VIDEOWRITER_PROP_DEPTH,
                                        cv2.CV_16U,
                                        cv2.VIDEOWRITER_PROP_IS_COLOR,
                                        0  # false
                                    ]
                                )
        else:
            share1 = cv2.VideoWriter(out1, 
                                    fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)), 
                                    params=[
                                        cv2.VIDEOWRITER_PROP_IS_COLOR,
                                        0  # false
                                    ]
                                )
            share2 = cv2.VideoWriter(out2, 
                                    fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)), 
                                    params=[
                                        cv2.VIDEOWRITER_PROP_IS_COLOR,
                                        0  # false
                                    ]
                                )
        share_creator = share_creator_lossy

    elif mode == 'lossless':
        share1 = cv2.VideoWriter(out1, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2),
                                                                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)))
        share2 = cv2.VideoWriter(out2, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2),
                                                                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)))
        share_creator = share_creator_lossless

    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            if ret==True:

                s1, s2 = share_creator.generate_shares(frame, high_res=high_res)
                # print(frame[0][0], s1[0][0], s2[0][0])
                # break
                # write the flipped frame
                share1.write(s1)
                share2.write(s2)
                cv2.imshow('Video',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        except cv2.error as e:
            pass

    # Release everything if job is finished
    cap.release()
    share1.release()
    cv2.destroyAllWindows()