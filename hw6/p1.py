import os
import sys
import numpy as np

from skimage import io

def scale(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

dir_path = sys.argv[1]  + '/'
filenames = [ os.path.join(dir_path, f) for f in os.listdir(dir_path) ]
imgs = np.concatenate([ io.imread(filename).reshape(1, -1) for filename in filenames ]).astype(np.float64)

avg_img = np.mean(imgs, axis=0)
norm_imgs = imgs - avg_img
U, s, V = np.linalg.svd(norm_imgs, full_matrices=False)

recog_file_path = os.path.join(dir_path,sys.argv[2]) 

img = io.imread(recog_file_path).reshape(1, -1).astype(np.float64)
img = img - avg_img
w = np.dot(img, V[:4].T).flatten()
recon_img = scale(np.sum([w[i] * V[i] for i in range(len(w))], axis=0, dtype=np.float64) + avg_img).reshape(600,600,3)

io.imsave('reconstruction.jpg', recon_img)
