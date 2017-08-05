import os, sys
from PIL import Image

size = 250, 250
infile = "xyz_longhair.jpg"

try:
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save("xyz_longhair_resize.jpg", "JPEG")
except IOError:
    print
    "cannot create thumbnail for '%s'" % infile
