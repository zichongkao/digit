import numpy as np

# color - "blackness" of pixel p (0 - 255)
# img_pix - number of imges in dataset with pixel p non-0
# img_total - total number of imges in dataset
def tfidf(color,img_pix,img_total):
	if img_pix==0:
		return 0
	else:
		return color * np.log(img_total/img_pix)

with open("C:/Users/kao/Google Drive/Desktop/website/digit/train.csv","r",encoding="utf-8") as f:
	header = f.readline()
	rawdata = [row.split(',') for row in f.read().splitlines()]

img_total = len(rawdata)

# prepare img_pix values
labels = [int(row.pop(0)) for row in rawdata]
data_matrix = [[int(elem) for elem in row] for row in rawdata]

# only interested in zero/non-zero distintion
bool_matrix = np.matrix(data_matrix, dtype=bool)
#bool_matrix = map(data_matrix)
img_pix_vec=bool_matrix.T.sum(1)

result = []
for img_vec in data_matrix:
	double_vec = zip(img_vec,img_pix_vec)
	result_row = [tfidf(x[0],x[1],img_total) for x in double_vec]
	result.append(result_row)