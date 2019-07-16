from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib

import matplotlib.pyplot as plt
import numpy as np
import sys

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-wnid', required=True, type=str, default=None)
parser.add_argument('-img_dir', required=True, type=str, default=None)
parser.add_argument('-max_images', required=False, type=int, default=1e7)
parser.add_argument('-progress_start', required=False, type=int, default=0)
parser.add_argument('-start_idx', required=False, type=int, default=1)



args = parser.parse_args()

wnid = args.wnid
max_images = args.max_images
img_dir = args.img_dir
progress_start = args.progress_start
start_idx = args.start_idx



# python get_images.py -wnid=n02391049 -img_dir=concepts/zebra3 -max_images=10

def url_to_image(url):

	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	return image


url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + wnid
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

str_soup = str(soup)
split_urls = str_soup.split('\r\n')
total_images = len(split_urls)

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

n_images = max_images if total_images > max_images else total_images


did_write = start_idx
progress = progress_start
#for progress in range(n_images):
while did_write < n_images + start_idx:
	progress += 1
	if progress > 10e8:
		break
	

	save_path = os.path.join(img_dir,'img' + str(did_write) + '.jpg')
	#if not split_urls[progress] == None:
	if not os.path.exists(save_path):
		try:
			I = url_to_image(split_urls[progress])
			if len(I.shape) == 3:
				
				did_write += 1
				if (did_write%20==0):
					print(did_write)
				
				cv2.imwrite(save_path,I)
				
		except:
				pass

print('wrote '+ str(did_write - 1) + ' of ' + str(n_images - start_idx) +' images')
print(f'ended on progress of {progress}')


