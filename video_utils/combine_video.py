import numpy as np
import cv2
from VSS.Lossless.numba_accelerated import share_combiner as share_combiner_lossless
from VSS.Lossy.numba_accelerated import share_combiner as share_combiner_lossy
def __combine_video__(share1_path:str = 'share1.avi', share2_path:str = 'share2.avi', output='output.avi', mode='lossless', high_res=True, verbose=True):
    share1 = cv2.VideoCapture(share1_path, apiPreference=cv2.CAP_FFMPEG)
    share2 = cv2.VideoCapture(share2_path, apiPreference=cv2.CAP_FFMPEG)

    # share1.set(fourcc, cv2.VideoWriter.fourcc(*'FFV1'))
    # share1.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    # share2.set(fourcc, cv2.VideoWriter.fourcc(*'FFV1'))
    # share2.set(cv2.CAP_PROP_CONVERT_RGB, 0)


    if (share1.isOpened() == False or share2.isOpened() == False): 
        raise Exception("Error opening video stream or file")
    fourcc = cv2.VideoWriter.fourcc(*'FFV1')
    combined = cv2.VideoWriter(output, fourcc, share1.get(cv2.CAP_PROP_FPS), (int(share1.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
                                                                               int(share1.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)))
    
    if mode == 'lossy':
        share_combiner = share_combiner_lossy
        if high_res:
            share1.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            share2.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    else:
        share_combiner = share_combiner_lossless




    while(share1.isOpened() and share2.isOpened()):
        ret1, frame1 = share1.read()
        ret2, frame2 = share2.read()

        # print(frame1[100:110, 100:110])
        # break
        if ret1==True and ret2 == True:
            if mode == 'lossy':
                frame1 = cv2.split(frame1)[0]
                frame2 = cv2.split(frame2)[0]

            combined_share = share_combiner.combine_shares(frame1, frame2, verbose=False, high_res=high_res)
            combined.write(combined_share)
            if verbose:
                cv2.imshow('Video', combined_share)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    share1.release()
    share1.release()
    cv2.destroyAllWindows()
