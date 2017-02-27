#!/usr/bin/env python


import sys
from PIL import Image

from basics import *
from networkBasics import *
from configuration import * 


def mergeImagesHorizontal(imageNames, finalImageName):

    images = map(Image.open, imageNames)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths) + 10 * len(images)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0] + 10

    new_im.save(finalImageName)


def mergeImagesVertical(imageNames, finalImageName):

    images = map(Image.open, imageNames)
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    max_height = sum(heights) + 10 * len(images)

    new_im = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    for im in images:
      new_im.paste(im, (0,y_offset))
      y_offset += im.size[1] + 10

    new_im.save(finalImageName)