import datetime
import math
import skimage.io as io
import pickle
import datetime
import time
import cv2
import os
import skimage
import random
from skimage import data
import random
import numpy as np 
import sklearn as sk
from sklearn import svm
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# def normalize()
def normalize(list, ranger): 
	l = np.array(list) 
	a = np.max(l)
	c = np.min(l)
	b = ranger[1]
	d = ranger[0]
	m = (b - d) / (a - c)
	pslope = (m * (l - c)) + d
	return pslope
ranger=np.zeros((2,1))
ranger[0]=0
ranger[1]=255

shape3d=(592,896,3)
a=592
b=896
shape2d=(592,896)
k=np.zeros((1,10))
j=12000
for i in range(10):
	k[0][i]=int(j)
	j=j+1000

image_dir="/home/sirshendu/Music/faces/"
file_names=[ os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

for f in file_names:
	img=cv2.imread(f);
	#cv2.imshow("Input image before any processing",img)
	#convert to grey scale.. cause methods in classroom only dealt with 2 Dimensional images to work 
	#cv2.waitKey(0)
	#break
	print(img.shape)
	imgh=img.copy()
	grayf = cv2.cvtColor(imgh, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Graeyscaled",grayf)
	cv2.waitKey(0)
	grayff=grayf.copy()
	copy1=grayff.copy()
	copy2=grayff.copy()
	copy3=grayff.copy()
	copy3=cv2.resize(copy3,(200,100))
	print("copy3 shape", copy3.shape)
	#break
	#cv2.imshow("resized image", copy3)
	#cv2.waitKey(0)
	#break
	#pca = PCA(n_components=30)
	# pca.fit(grayff)
	# pca.transform
	for i in range(10):
		cat=int(k[0][i]
		=0.0
		sigma=float(1/cat)
		maze=np.zeros((int(k[0][i]),100))
		# for c in range(int(k[0][i])):
		# 	for d in range(a):
		# 		#maze[c][d]=np.random.normal(mu,sigma,1)
		# 		num=random.randint(0,)
		for d in range(100):
			num=random.randint(0,((cat-1))
			maze[num][d]=1
			num2=random.randint(0,1)
			if num2==0:
				maze[num][d]=maze[num][d]*1
			else:
				maze[num][d]=maze[num][d]*(-1)
		#img_new=np.matmul(maze,grayff)
		img11=maze.copy()
		img12=maze.copy()
		dummy=np.transpose(img11)
		dummy2=np.matmul(dummy,img12)
		img_new=np.matmul(dummy2,copy3)
		cv2.imshow(" Reconstructed but not resized ", img_new)
		cv2.waitKey(0)
		img_new=cv2.resize(img_new,(896,592))
		cv2.imshow("Reconstructed and resized ",img_new)
		cv2.waitKey(0)
		for k in range(592):
			img_new[i]=normalize(img_new[i],ranger)
		cv2.imshow("Normalized afterall",img_new)
		cv2.waitKey(0)
	break



	#cv2.imshow("gray scaled", grayf)
	#cv2.waitKey(0)
	print(grayf.shape)
	break
##############   as we can see ( and reasonably so) , #################################
########################## the image needs to be normalized ##########################