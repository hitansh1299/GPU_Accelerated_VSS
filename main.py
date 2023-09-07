from video_utils import combine_video
from video_utils import generate_video
import numpy as np
if __name__ == '__main__':

    # LOSSY VIDEO LOW RES
    generate_video.create_shares(input='Sample.mp4', 
                                 out1='share1_lossy_8_bit.avi',
                                 out2='share2_lossy_8_bit.avi',
                                 mode='lossy',
                                 high_res=False)

    combine_video.__combine_video__(
        share1_path='share1_lossy_8_bit.avi',
        share2_path='share2_lossy_8_bit.avi',
        output='output_lossy.avi',
        mode='lossy',
        high_res=False
    )

    # LOSSY VIDEO HIGH RES
    # generate_video.create_shares(input='Sample.mp4', 
    #                              out1='share1_lossy_16_bit.avi',
    #                              out2='share2_lossy_16_bit.avi',
    #                              mode='lossy',
    #                              high_res=True)

    # combine_video.__combine_video__(
    #     share1_path='share1_lossy_16_bit.avi',
    #     share2_path='share2_lossy_16_bit.avi',
    #     output='output_lossy_high_res.avi',
    #     mode='lossy',
    #     high_res=True
    # )

    # LOSSLESS VIDEO
    # generate_video.create_shares(input='Sample.mp4', 
    #                              out1='share1_lossless.avi',
    #                              out2='share2_lossless.avi',
    #                              mode='lossless')

    # combine_video.__combine_video__(
    #     share1_path='share1_lossless.avi',
    #     share2_path='share2_lossless.avi',
    #     output='output.avi',
    #     mode='lossless'
    # )