import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in xrange(0,n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

def print_images(output, img_path):
    seg_path = img_path.replace("jpg", "png")

    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    out_img = Image.fromarray(out_img)
    out_img.putpalette(getpallete(256))
    out_img.save(seg_path)

    # Display input
    print "Input Image:"
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()

    # Display output
    print "Output Image:"
    img_out = mpimg.imread(seg_path)
    plt.imshow(img_out)
    plt.show()
    
