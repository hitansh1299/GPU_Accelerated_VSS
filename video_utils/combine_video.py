import numpy as np
import cv2
from VSS.Lossless.numba_accelerated import share_combiner as share_combiner_lossless
from VSS.Lossy.numba_accelerated import share_combiner as share_combiner_lossy
def __combine_video__(share1_path:str = 'share1.avi', share2_path:str = 'share2.avi', output='output.avi', mode='lossless', verbose=True):
    share1 = cv2.VideoCapture(share1_path)
    share2 = cv2.VideoCapture(share2_path)
    if (share1.isOpened() == False or share2.isOpened() == False): 
        raise Exception("Error opening video stream or file")
    fourcc = cv2.VideoWriter.fourcc(*'HFYU')
    combined = cv2.VideoWriter(output, fourcc, share1.get(cv2.CAP_PROP_FPS), (int(share1.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
                                                                               int(share1.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)))

    share_combiner = share_combiner_lossy if mode == 'lossy' else share_combiner_lossless
 
    while(share1.isOpened() and share2.isOpened()):
        ret1, frame1 = share1.read()
        ret2, frame2 = share2.read()
        # print(frame1.shape, frame2.shape)
        # print(frame1[5:10, 5:10])
        # break
        if ret1==True and ret2 == True:
            if mode == 'lossy':
                frame1 = cv2.split(frame1)[0]
                frame2 = cv2.split(frame2)[0]
            # print(frame1.shape)
            combined_share = share_combiner.combine_shares(frame1, frame2, verbose=False)
            # print(frame1.shape)
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
