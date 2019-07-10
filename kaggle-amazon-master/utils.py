from haze_removal import getRecoverScene
from fastai.vision import *
import glob
import sys
from concurrent.futures import ThreadPoolExecutor , wait , ALL_COMPLETED
from pathlib import Path
import cv2
from concurrent import futures
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def remove_haze1(input_path, output_path):
	if not output_path.exists():
		output_path.mkdir()
	
	input_images_list = glob.glob(str(input_path)+'/*.jpg')
	for image_path in input_images_list:
		img = cv2.imread(image_path)
		dehazed_img1 = getRecoverScene(img, refine=True)
		file_name = Path(image_path).name
		output_file_path = output_path/file_name
		cv2.imwrite(str(output_file_path) , dehazed_img1)
		print(f"wrtien {str(output_file_path)}")

def remove_haze_from_image(image_path , output_path):
	img = cv2.imread(image_path)
	dehazed_img1 = getRecoverScene(img, refine=True)
	file_name = Path(image_path).name
	output_file_path = output_path/file_name
	cv2.imwrite(str(output_file_path) , dehazed_img1)
	print(f"wrtien {str(output_file_path)}")


def remove_haze(input_path, output_path):
	if not output_path.exists():
		output_path.mkdir()
	input_images_list = glob.glob(str(input_path)+'/*.jpg')
	output_images_list = [os.path.basename(x) for x in glob.glob(str(output_path)+'/*.jpg')]
	executor = ThreadPoolExecutor(max_workers=4)
	jobs = []
	for image_path in input_images_list:
		if Path(image_path).name not in output_images_list:
			jobs.append(executor.submit(remove_haze_from_image , image_path , output_path))
		else:
			print(f"alredy done {image_path}")
	for job in futures.as_completed(jobs):
		job.result()

def transpose(x):
	return x.transpose(1,2)

def preprocess_data(path,folder = "train-jpg"):
	#removing haze and storing in output_path
	#output_path = input_path/"noHaze_train-jpg" 
	#remove_haze(input_path/"train-jpg" , output_path)
	np.random.seed(42)
	
	#transpose
	r_transpose = TfmPixel(transpose)()
	r_transpose.p = 0.5
	xtra_tfms = [ r_transpose ]
	tfms = get_transforms(do_flip = True , flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0. ,xtra_tfms=xtra_tfms)
	np.random.seed(42)
	print(folder)
	src = (ImageList.from_csv(path, 'train_v2.csv', folder=folder, suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))
	data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))
	return data



if __name__ == "__main__":
	input_path = Path("./data")
	preprocess_data(input_path)