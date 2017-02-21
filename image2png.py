#!/usr/bin/python

import sys
import numpy as np
import png


if len(sys.argv) != 3:
  print "Usage: <in file> <out file>"
  exit()

img = np.loadtxt(sys.argv[1], delimiter=',',dtype=np.uint16)

with open(sys.argv[2], 'wb') as f:
    writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=16, greyscale=True)
    img_list = img.tolist()
    writer.write(f, img_list)

