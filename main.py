from video_utils import combine_video
from video_utils import generate_video
if __name__ == '__main__':
    generate_video.create_shares(input='Sample.mp4', 
                                 out1='share1_lossy.avi',
                                 out2='share2_lossy.avi',
                                 mode='lossy')
    
    combine_video.__combine_video__(
        share1_path='share1_lossy.avi',
        share2_path='share2_lossy.avi',
        output='output_lossy.avi',
        mode='lossy'
    )

    ## LOSSLESS VIDEO
    # generate_video.create_shares(input='Sample.mp4', 
    #                              out1='share1.avi',
    #                              out2='share2.avi',
    #                              mode='lossless')

    # combine_video.__combine_video__(
    #     share1_path='share1.avi',
    #     share2_path='share2.avi',
    #     output='output.avi',
    #     mode='lossless'
    # )