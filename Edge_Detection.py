import numpy as np
import math
import matplotlib.pyplot as plt

# input image

filename = '/Users/shivani/Downloads/downloads/test3.img'
output = '/Users/shivani/Downloads/outputCV.img'
output1 = '/Users/shivani/Downloads/BTimg.img'

with open(filename , 'rb') as in_file:
    with open(output, 'wb') as out_file:
        out_file.write(in_file.read()[512:])    # image header to remove size is known for the image

fo = open(output, 'rb')

Output512 = np.fromfile(output, dtype='uint8', sep="")
Output512 = Output512.reshape([512, 512])       # size is known for the image

with open(output , 'rb') as out_file1:
    myArr = bytearray(out_file1.read())

img = np.copy(Output512)
plt.imshow(img,cmap='gray')
plt.show()

# define sigma and other parameters

def parametersForFilter(sigma):
    # define the size of the filter
    size = int(2 * (np.ceil(4 * sigma)) + 1)

    # define the values in the filter
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1), np.arange(-size / 2 + 1, size / 2 + 1))

    # normalize the filter values
    normalize = 1 / (2.0 * np.pi * sigma ** 2)

    # LoG filter
    LOGfilter = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normalize

    # filter size
    filter_size= LOGfilter.shape[0]

    return LOGfilter, filter_size

sigma = 5

LOGfilter , filter_size = parametersForFilter(sigma)

# add padding of zeroes to the image as per sigma

padding = math.floor(filter_size/2)

newImage = np.zeros(shape=(img.shape[0] + 2* padding, img.shape[1] + 2 * padding))

for i in range(0,newImage.shape[0],1):
    for j in range(0,newImage.shape[1],1):
        if i <= padding or j <= padding :
            continue
        else:
            if i - padding < 512 and j - padding < 512:
                newImage[i][j] = img[i - padding][j - padding]

logarr = np.zeros_like(newImage, dtype=float)

# apply filter
for i in range(newImage.shape[0]-(filter_size - 1)):
    for j in range(newImage.shape[1]-(filter_size - 1)):
        if i + filter_size  < 512 and j + filter_size  < 512:
            window = newImage[i: i + filter_size, j:j + filter_size] * LOGfilter
            logarr[i + padding, j + padding] = np.sum(window)

logarr = logarr.astype(np.int64, copy=False)

zero_crossing = np.zeros_like(logarr)

#find zero crossings
for i in range(logarr.shape[0]):
    for j in range(logarr.shape[1]):
        if logarr[i][j] == 0:
            # to check for vertical or horizontal
            if (logarr[i][j - 1] < 0 and logarr[i][j + 1] > 0) or (logarr[i][j - 1] < 0 and logarr[i][j + 1] < 0) or (
                    logarr[i - 1][j] < 0 and logarr[i + 1][j] > 0) or (logarr[i - 1][j] > 0 and logarr[i + 1][j] < 0):
                zero_crossing[i][j] = 255
        if logarr[i][j] < 0:
            if (logarr[i][j - 1] > 0) or (logarr[i][j + 1] > 0) or (logarr[i - 1][j] > 0) or (logarr[i + 1][j] > 0):
                zero_crossing[i][j] = 255

# for sigma - delta sigma   5-0.5 = 4.5
# zero_crossing have the edge map for previous sigma
# newImage have the padded image data to refer for zero crossings
# convolve around the pixel and 8-neighbor whose value is 255 in edge map



# fuction to convolve image with given edgemap
def convolveImage(zero_crossing,padding,filter_size_4_5,LOGfilter4_5,logArr4_5,newImage):

    for i in range(zero_crossing.shape[0]):
        for j in range(zero_crossing.shape[1]):
            if zero_crossing[i][j] == 255 and (i + filter_size_4_5) < zero_crossing.shape[0] and (j + filter_size_4_5) \
                    < zero_crossing.shape[1] and (i + padding) < zero_crossing.shape[0] and (j + padding) < zero_crossing.shape[1]:
                # convolve around the same pixel in the original image and its 8-neighbors
                window = newImage[i: i + filter_size_4_5, j:j + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i + padding, j + padding] = np.sum(window)
                # 8-neighbors

                # i-1 ; j-1
                window = newImage[i-1: i-1 + filter_size_4_5, j-1:j-1 + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i-1 + padding, j-1 + padding] = np.sum(window)

                # i-1 ; j
                window = newImage[i - 1: i - 1 + filter_size_4_5, j:j + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i - 1 + padding, j + padding] = np.sum(window)

                # i-1 ; j+1
                window = newImage[i - 1: i - 1 + filter_size_4_5, j + 1:j + 1 + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i - 1 + padding, j + 1 + padding] = np.sum(window)

                # i ; j-1
                window = newImage[i : i + filter_size_4_5, j - 1:j - 1 + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i + padding, j - 1 + padding] = np.sum(window)

                # i-1 ; j+1
                window = newImage[i - 1: i - 1 + filter_size_4_5, j + 1:j + 1 + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i - 1 + padding, j + 1 + padding] = np.sum(window)

                # i+1 ; j-1
                window = newImage[i + 1: i + 1 + filter_size_4_5, j - 1:j - 1 + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i + 1 + padding, j - 1 + padding] = np.sum(window)

                # i+1 ; j
                window = newImage[i + 1: i + 1 + filter_size_4_5, j:j + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i + 1 + padding, j + padding] = np.sum(window)

                # i+1 ; j+1
                window = newImage[i + 1: i + 1 + filter_size_4_5, j + 1:j + 1 + filter_size_4_5] * LOGfilter4_5
                logArr4_5[i + 1 + padding, j + 1 + padding] = np.sum(window)

    logArr4_5 = logArr4_5.astype(np.int64, copy=False)

    zero_crossing_4_5 = np.zeros_like(logArr4_5)

    #find zero crossings
    for i in range(logArr4_5.shape[0]):# - (filter_size - 1)):
        for j in range(logArr4_5.shape[1]):#- (filter_size - 1)):
            if logArr4_5[i][j] == 0:
                # to check for vertical or horizontal
                if (logArr4_5[i][j - 1] < 0 and logArr4_5[i][j + 1] > 0) or (logArr4_5[i][j - 1] < 0 and logArr4_5[i][j + 1] < 0) or (
                      logArr4_5[i - 1][j] < 0 and logArr4_5[i + 1][j] > 0) or (logArr4_5[i - 1][j] > 0 and logArr4_5[i + 1][j] < 0):
                    zero_crossing_4_5[i][j] = 255
            if logArr4_5[i][j] < 0:
                if (logArr4_5[i][j - 1] > 0) or (logArr4_5[i][j + 1] > 0) or (logArr4_5[i - 1][j] > 0) or (logArr4_5[i + 1][j] > 0):
                    zero_crossing_4_5[i][j] = 255

    return zero_crossing_4_5

#plot the image
def plotedges(zero_crossing,sigma):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(zero_crossing, cmap='gray')
    string = '\u03C3 = '
    string += (str(sigma))
    a.set_title(string)
    plt.show()


sigma = 4.5
LOGfilter4_5, filter_size_4_5 = parametersForFilter(sigma)
logArr4_5 = np.zeros_like(newImage, dtype=float)
zero_crossing_4_5 = convolveImage(zero_crossing,padding,filter_size_4_5,LOGfilter4_5,logArr4_5,newImage)

sigma = 4
LOGfilter4, filter_size_4 = parametersForFilter(sigma)
logArr4 = np.zeros_like(newImage, dtype=float)
zero_crossing_4 = convolveImage(zero_crossing_4_5,padding,filter_size_4,LOGfilter4,logArr4,newImage)

sigma = 3.5
LOGfilter3_5, filter_size_3_5 = parametersForFilter(sigma)
logArr3_5 = np.zeros_like(newImage, dtype=float)
zero_crossing_3_5 = convolveImage(zero_crossing_4,padding,filter_size_3_5,LOGfilter3_5,logArr3_5,newImage)

sigma = 3
LOGfilter3, filter_size_3 = parametersForFilter(sigma)
logArr3 = np.zeros_like(newImage, dtype=float)
zero_crossing_3 = convolveImage(zero_crossing_3_5,padding,filter_size_3,LOGfilter3,logArr3,newImage)

sigma = 2.5
LOGfilter2_5, filter_size_2_5 = parametersForFilter(sigma)
logArr2_5 = np.zeros_like(newImage, dtype=float)
zero_crossing_2_5 = convolveImage(zero_crossing_3,padding,filter_size_2_5,LOGfilter2_5,logArr2_5,newImage)

sigma = 2
LOGfilter2, filter_size_2 = parametersForFilter(sigma)
logArr2 = np.zeros_like(newImage, dtype=float)
zero_crossing_2 = convolveImage(zero_crossing_2_5,padding,filter_size_2,LOGfilter2,logArr2,newImage)

sigma = 1.5
LOGfilter1_5, filter_size_1_5 = parametersForFilter(sigma)
logArr1_5 = np.zeros_like(newImage, dtype=float)
zero_crossing_1_5 = convolveImage(zero_crossing_2,padding,filter_size_1_5,LOGfilter1_5,logArr1_5,newImage)

sigma = 1
LOGfilter1, filter_size_1 = parametersForFilter(sigma)
logArr1 = np.zeros_like(newImage, dtype=float)
zero_crossing_1 = convolveImage(zero_crossing_1_5,padding,filter_size_1,LOGfilter1,logArr1,newImage)

sigma = 0.5
LOGfilter0_5, filter_size_0_5 = parametersForFilter(sigma)
logArr0_5 = np.zeros_like(newImage, dtype=float)
zero_crossing_0_5 = convolveImage(zero_crossing_1,padding,filter_size_0_5,LOGfilter0_5,logArr0_5,newImage)

# sigma = 5
newcross = zero_crossing

for i in range(newcross.shape[0]):
    for j in range(newcross.shape[1]):
        if newcross[i][j] != 0 :
            newcross[i][j] = 255

newcross1 = np.zeros(shape=(512,512))

for i in range(newcross1.shape[0]):
    for j in range(newcross1.shape[1]):
        newcross1[i][j] = newcross[i + padding][j + padding]

plotedges(newcross1,5)

# sigma = 4
newcross = zero_crossing + zero_crossing_4_5 + zero_crossing_4

for i in range(newcross.shape[0]):
    for j in range(newcross.shape[1]):
        if newcross[i][j] != 0 :
            newcross[i][j] = 255

newcross1 = np.zeros(shape=(512,512))

for i in range(newcross1.shape[0]):
    for j in range(newcross1.shape[1]):
        newcross1[i][j] = newcross[i + padding][j + padding]

plotedges(newcross1,4)

# sigma = 3
newcross = zero_crossing + zero_crossing_4 + zero_crossing_4_5 + zero_crossing_3_5 + zero_crossing_3

for i in range(newcross.shape[0]):
    for j in range(newcross.shape[1]):
        if newcross[i][j] != 0 :
            newcross[i][j] = 255

newcross1 = np.zeros(shape=(512,512))

for i in range(newcross1.shape[0]):
    for j in range(newcross1.shape[1]):
        newcross1[i][j] = newcross[i + padding][j + padding]

plotedges(newcross1,3)

# sigma = 2
newcross = zero_crossing + zero_crossing_4 + zero_crossing_4_5 + zero_crossing_3_5 + zero_crossing_3 + zero_crossing_2_5 + zero_crossing_2

for i in range(newcross.shape[0]):
    for j in range(newcross.shape[1]):
        if newcross[i][j] != 0 :
            newcross[i][j] = 255

newcross1 = np.zeros(shape=(512,512))

for i in range(newcross1.shape[0]):
    for j in range(newcross1.shape[1]):
        newcross1[i][j] = newcross[i + padding][j + padding]

plotedges(newcross1,2)

# sigma = 1
newcross = zero_crossing + zero_crossing_4 + zero_crossing_4_5 + zero_crossing_3_5 + zero_crossing_3 + zero_crossing_2_5 + zero_crossing_2 + zero_crossing_1_5 + zero_crossing_1

for i in range(newcross.shape[0]):
    for j in range(newcross.shape[1]):
        if newcross[i][j] != 0 :
            newcross[i][j] = 255

newcross1 = np.zeros(shape=(512,512))

for i in range(newcross1.shape[0]):
    for j in range(newcross1.shape[1]):
        newcross1[i][j] = newcross[i + padding][j + padding]

plotedges(newcross1,1)