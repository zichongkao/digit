import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open("C:/Users/kao/Google Drive/Desktop/website/digit/train.csv","r",encoding="utf-8") as f:
	header = f.readline()
	rawdata = [row.split(',') for row in f.read().splitlines()]

img_total = len(rawdata)

# prepare img_pix values
labels = [int(row.pop(0)) for row in rawdata]
data_matrix = np.array(rawdata,dtype=int)

# only interested in zero/non-zero distinction
bool_matrix = data_matrix.astype(bool)
img_pix_vec=bool_matrix.T.sum(1)
img_pix_mat=np.array([img_pix_vec]*img_total,dtype=int)

result = data_matrix * np.log(np.divide(img_total,img_pix_mat))
# img_pix_mat 0's lead to Nan's in the result. Convert these to 0
result[np.isnan(result)] = 0

# for plotting
# select image in dataset
index = 0

# plot the tfidf_img
tfidf_img = result[index].reshape(28,28)
plt.imshow(tfidf_img,cmap=cm.Greys_r,interpolation="nearest")
plt.show()

# plot the original img
original_img = np.array(data_matrix[index]).reshape(28,28)
plt.imshow(original_img,cmap=cm.Greys_r,interpolation="nearest")
plt.show()