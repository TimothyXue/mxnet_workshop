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

def get_results(exector,pallete,img,seg):
    tic = time.time()
    exector.forward(is_train=False)
    print "Time taken for forward pass: {:.3f} milli sec".format((time.time()-tic)*1000)

    print "Postprocessing results to display output..."
    output = exector.outputs[0]
    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    out_img = Image.fromarray(out_img)
    out_img.putpalette(pallete)
    out_img.save(seg)

    # Display input
    print "Input Image:"
    img_in = mpimg.imread(img)
    imgplot = plt.imshow(img_in)
    plt.show()

    # Display output
    print "Output Image:"
    img_out = mpimg.imread(seg)
    imgplot = plt.imshow(img_out)
    plt.show()
