import os
from os import path
import scipy as sp
import itertools as it

from PIL import Image, ImageOps
from glob import glob

path_src = "./data/txt"
path_out = "./data/img"
path_stamp = "./data/bad_img"
abspath_src = path.abspath(path_src)
abspath_out = path.abspath(path_out)


def flip_and_rotator(marked, axis=None, angle=0):
  
  flip_axis = {'vertical' : lambda x: x[:,::-1],
               'horizontal': lambda x:x[::-1,:],
               'transposed': lambda x:x[::-1,::-1]}
  
  if axis in flip_axis.key():
    flipped = flip_axis[axis](marked)
    
  else axis not in flip_axis.key():
    raise Exception('The given axis is inappropriate.')
  
  rotated = sp.ndimage.rotate(flipped, angle, cval=255, mode='nearest', reshape=0)
  
  return rotated


file_paths = [path.join(abspath_src, name) for name in os.listdir(path_src) if path.isfile(path.join(abspath_src, name))]

bad_name = sorted(glob(path_out + '/' + '*.png'))
bad_list = list((sp.misc.imresize(sp.misc.imread(bad), size=(100,100))) for bad in bad_name)

mark_list = bad_list
axis_list = ['vertical', 'horizontal', 'transposed']
angle_gen = range(0, 360, 30)

raw_list = [flip_and_rotator(mark, axis=axis, angle=angle) for mark, axis, angle in it.product(mark_list, axis_list, angle_gen)]
angle_cnt = len(raw_list)/len(bad_list)

# The results
print('Augmentated pattern : %s' % len(raw_list))

for idx, val in enumerate(raw_list):
  val = 255 - val
  img = Image.fromarray(val)
  img = ImageOps.invert(img)
  #img.show()
  img_filename = bad_list[int(idx/angle_cnt)].replace('img', path_stamp[-7:])[:-4] + '_' + str(idx) + '.png'
  img.save(img_filename)
  
            
            
            
            
