# from multiprocessing import shared_memory
import time

from image import ImageUtil, ImageLoader
import mmap
import array
import sys
import numpy as np
import cv2

data_dir = '/home/x/Workspace/sdetector/data/act2'

# required size
# text buffer: 4096
# image buffer : 1280 x 720 x 3 =

image_size = 1280*720*3
shm_size = 4096 + 1280*720*3
# shm_a = shared_memory.SharedMemory(create=True, name='sdetector_shm', size=shm_size)
# buf = shm_a.buf

if sys.argv[1] == 'writer':
    filelist='data/filelist-all.txt'
    print('reading files..')
    images, filelist=ImageLoader.read_image_from_filelist(filelist)
    for i in range(len(images)):
        print(f'writing to mmap : {i}')
        with open('shared.data', 'wb+') as fd:
            # fd.write(b'{:08d}'.format(i))
            fd.write(images[i].tobytes())
        time.sleep(1)
else:
    with open('shared.data', 'rb+') as fd:
        mm = mmap.mmap(fd.fileno(), mmap.ACCESS_DEFAULT)
        while True:
            print('reading')
            data = mm.read(image_size)
            image=np.frombuffer(data, dtype=np.uint8, count=3)
            cv2.imshow('image', image)
            if cv2.waitKey(50) == ord('q'):
                break
            mm.seek(0)
