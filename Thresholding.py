import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as signal
import math


filename = '/Users/shivani/Downloads/test2.img'
output = '/Users/shivani/Downloads/outputCV.img'


with open(filename , 'rb') as in_file:
    with open(output, 'wb') as out_file:
        out_file.write(in_file.read()[512:])

fo = open(output, 'rb')

#below file will have 512 X 512 image data
Output512 = np.fromfile(output, dtype='uint8', sep="")
Output512 = Output512.reshape([512, 512])

plt.imshow(Output512, cmap='gray')
plt.show()

tempArr = np.copy(Output512)


####### 1. PEAKINESS DETECTION #########
# assign measure of goodness to determine whether to select a combination of peak valleys or not

#generate histogram for the respective image
#a = np.histogram(Output512,bins=1,range=None, normed=None, weights=None, density=None)
#plt.hist(Output512,bins=40)
#plt.show()


histdata = [0] * 256

for x in range(0,512,1):
    for y in range(0,512,1):
        histdata[Output512[x][y]] = histdata[Output512[x][y]] + 1

#selection criteria for peaks
peaks = []
for item in histdata:
    if item >= 2500 : peaks.append(item)

#selection criteria for valleys
valleys = []
for item in histdata:
    if item <= 300 : valleys.append(item)

peakIDX =[]
valleyIDX = []
for item in peaks:
    peakIDX.append(histdata.index(item))

for item in valleys:
    valleyIDX.append(histdata.index(item))


#Measure of goodness criteria
# 1. peaks should be well seperated. distance >= 70
# 2. peaks are dominant as per our criteria
# 3. valleys selected are low enough
# 4. valley should be close to the center between the peaks. +- 5
# 5. calculate measure of goodness =

peaks2 = peaks

finalT = 0
finalp1 = 0
finalp2 = 0
MOG = 0
MOGold = 0
for peak1 in peaks:
    for peak2 in peaks2:
        if peak1 == peak2 :
            pass
        else :
            if abs(histdata.index(peak1) - histdata.index(peak2)) >= 70:
                for valley in valleys:
                    if histdata.index(peak1) <= histdata.index(valley) and histdata.index(peak2) >= histdata.index(valley):
                        centre = math.ceil((histdata.index(peak1) + histdata.index(peak2)) / 2)
                        if histdata.index(valley) <= (centre + 20) and histdata.index(valley) >= (centre - 20):
                            MOG = (abs(histdata.index(peak1) - histdata.index(peak2)) * (peak1 + peak2)/2) / ( histdata.index(valley) )

                            if MOG >= 4000 and MOGold < MOG:
                                finalT = histdata.index(valley)
                                finalp1 = histdata.index(peak1)
                                finalp2 = histdata.index(peak2)
                            MOGold = MOG

if finalT != 0:
    for x in range(0,512,1):
        for y in range(0,512,1):
            if Output512[x][y] >= finalT:
                Output512[x][y] = 255
            else:
                Output512[x][y] = 0


peakinessFinalImage = np.copy(Output512)


####### 2. ITERATIVE #########

Output512 = np.copy(tempArr)

mean = math.ceil(np.mean(Output512))

# define intensity 2D list
masterlist = []
list1 = []
list2 = []
newTempArr = np.copy(Output512)

#plt.imshow(newTempArr, cmap='gray')
#plt.show()

for x in range(0,512,1):
    for y in range(0,512,1):
        if newTempArr[x][y] < mean:
            list1.append(newTempArr[x][y])
        elif newTempArr[x][y] >= mean:
            list2.append(newTempArr[x][y])
masterlist.append(list1)
masterlist.append(list2)

newMean = math.ceil(np.mean(list1))
newMean1 = math.ceil(np.mean(list2))

Tnext = ( newMean + newMean1 ) / 2

oldMean = newMean
oldMean1 = newMean1
Tprev = Tnext
count = 0
while True:
    list1 = []
    list2 = []
    for x in range(0,512,1):
        for y in range(0,512,1):
            if newTempArr[x][y] < Tnext:
                list1.append(newTempArr[x][y])
            else:
                if newTempArr[x][y] >= Tnext:
                    list2.append(newTempArr[x][y])
    newMean = math.ceil(np.mean(list1))
    newMean1 = math.ceil(np.mean(list2))
    Tprev = Tnext
    Tnext = ( newMean + newMean1 ) / 2

    count = count + 1

    if Tnext == Tprev:
        break

for x in range(0,512,1):
    for y in range(0,512,1):
        if newTempArr[x][y] < math.ceil(Tnext):
            newTempArr[x][y] = 0
        elif newTempArr[x][y] >= math.ceil(Tnext):
            newTempArr[x][y] = 255

iterativeOutputArr = np.copy(newTempArr)

####### 3. ADAPTIVE THRESHOLDING #########

Output512 = np.copy(tempArr)

# built dataset for histogram data


histdata = [0] * 256

for x in range(0,512,1):
    for y in range(0,512,1):
        histdata[Output512[x][y]] = histdata[Output512[x][y]] + 1


#divide the image into 256 X 256 blocks i.e into 4 sub-images

subImage1 = np.zeros((256,256),dtype='uint8')
subImage2 = np.zeros((256,256),dtype='uint8')
subImage3 = np.zeros((256,256),dtype='uint8')
subImage4 = np.zeros((256,256),dtype='uint8')

i = 0
j = 0
for x in range(0,256,1):
    for y in range(0,256,1):
        subImage1[i][j] = Output512[x][y]
        j = j + 1
    j = 0
    i = i + 1

i = 0
j = 0
for x in range(256,512,1):
    for y in range(0,256,1):
        subImage2[i][j] = Output512[x][y]
        j = j + 1
    j = 0
    i = i + 1

i = 0
j = 0
for x in range(0,256,1):
    for y in range(256,512,1):
        subImage3[i][j] = Output512[x][y]
        j = j + 1
    j = 0
    i = i + 1

i = 0
j = 0
for x in range(256,512,1):
    for y in range(256,512,1):
        subImage4[i][j] = Output512[x][y]
        j = j + 1
    j = 0
    i = i + 1

# Iterative for subImage1

#built hist-data for subimg1
histdataIMG1 = [0] * 256


for x in range(0,256,1):
    for y in range(0,256,1):
        histdataIMG1[subImage1[x][y]] = histdataIMG1[subImage1[x][y]] + 1

mean = math.ceil(np.mean(subImage1))
list1 = []
list2 = []
newTempArr = np.copy(subImage1)

Tnext = mean

for x in range(0,256,1):
    for y in range(0,256,1):
        if newTempArr[x][y] < Tnext:
            list1.append(newTempArr[x][y])
        else:
            if newTempArr[x][y] >= Tnext:
                list2.append(newTempArr[x][y])

newMean = math.ceil(np.mean(list1))
newMean1 = math.ceil(np.mean(list2))

Tnext = ( newMean + newMean1 ) / 2

oldMean = newMean
oldMean1 = newMean1
Tprev = Tnext
count = 0
while True:
    list1 = []
    list2 = []
    for x in range(0,256,1):
        for y in range(0,256,1):
            if newTempArr[x][y] < Tnext:
                list1.append(newTempArr[x][y])
            else:
                if newTempArr[x][y] >= Tnext:
                    list2.append(newTempArr[x][y])
    newMean = math.ceil(np.mean(list1))
    newMean1 = math.ceil(np.mean(list2))
    Tprev = Tnext
    Tnext = ( newMean + newMean1 ) / 2

    count = count + 1

    if Tnext == Tprev:
        break

for x in range(0,256,1):
    for y in range(0,256,1):
        if newTempArr[x][y] < math.ceil(Tnext):
            newTempArr[x][y] = 0
        elif newTempArr[x][y] >= math.ceil(Tnext):
            newTempArr[x][y] = 255

#plt.imshow(newTempArr, cmap='gray')
#plt.show()

opsubImage1 = np.copy(newTempArr)

# Iterative for subImage2

# built hist-data for subimg2
histdataIMG2 = [0] * 256

for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        histdataIMG2[subImage2[x][y]] = histdataIMG2[subImage2[x][y]] + 1

mean = math.ceil(np.mean(subImage2))
list1 = []
list2 = []
newTempArr = np.copy(subImage2)

Tnext = mean

for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        if newTempArr[x][y] < Tnext:
            list1.append(newTempArr[x][y])
        else:
            if newTempArr[x][y] >= Tnext:
                list2.append(newTempArr[x][y])

newMean = math.ceil(np.mean(list1))
newMean1 = math.ceil(np.mean(list2))

Tnext = (newMean + newMean1) / 2

oldMean = newMean
oldMean1 = newMean1
Tprev = Tnext
count = 0
while True:
    list1 = []
    list2 = []
    for x in range(0, 256, 1):
        for y in range(0, 256, 1):
            if newTempArr[x][y] < Tnext:
                list1.append(newTempArr[x][y])
            else:
                if newTempArr[x][y] >= Tnext:
                    list2.append(newTempArr[x][y])
    newMean = math.ceil(np.mean(list1))
    newMean1 = math.ceil(np.mean(list2))
    Tprev = Tnext
    Tnext = (newMean + newMean1) / 2

    count = count + 1

    if Tnext == Tprev:
        break


for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        if newTempArr[x][y] < math.ceil(Tnext):
            newTempArr[x][y] = 0
        elif newTempArr[x][y] >= math.ceil(Tnext):
            newTempArr[x][y] = 255

#plt.imshow(newTempArr, cmap='gray')
#plt.show()

opsubImage2 = np.copy(newTempArr)

# Iterative for subImage3

# built hist-data for subimg3
histdataIMG3 = [0] * 256

for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        histdataIMG3[subImage3[x][y]] = histdataIMG3[subImage3[x][y]] + 1

mean = math.ceil(np.mean(subImage3))
list1 = []
list2 = []
newTempArr = np.copy(subImage3)

Tnext = mean

for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        if newTempArr[x][y] < Tnext:
            list1.append(newTempArr[x][y])
        else:
            if newTempArr[x][y] >= Tnext:
                list2.append(newTempArr[x][y])

newMean = math.ceil(np.mean(list1))
newMean1 = math.ceil(np.mean(list2))

Tnext = (newMean + newMean1) / 2

oldMean = newMean
oldMean1 = newMean1
Tprev = Tnext
count = 0
while True:
    list1 = []
    list2 = []
    for x in range(0, 256, 1):
        for y in range(0, 256, 1):
            if newTempArr[x][y] < Tnext:
                list1.append(newTempArr[x][y])
            else:
                if newTempArr[x][y] >= Tnext:
                    list2.append(newTempArr[x][y])
    newMean = math.ceil(np.mean(list1))
    newMean1 = math.ceil(np.mean(list2))
    Tprev = Tnext
    Tnext = (newMean + newMean1) / 2

    count = count + 1

    if Tnext == Tprev:
        break


for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        if newTempArr[x][y] < math.ceil(Tnext):
            newTempArr[x][y] = 0
        elif newTempArr[x][y] >= math.ceil(Tnext):
            newTempArr[x][y] = 255

#plt.imshow(newTempArr, cmap='gray')
#plt.show()

opsubImage3 = np.copy(newTempArr)

# Iterative for subImage4

# built hist-data for subimg4
histdataIMG4 = [0] * 256

for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        histdataIMG4[subImage4[x][y]] = histdataIMG4[subImage4[x][y]] + 1

mean = math.ceil(np.mean(subImage4))
list1 = []
list2 = []
newTempArr = np.copy(subImage4)

Tnext = mean

for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        if newTempArr[x][y] < Tnext:
            list1.append(newTempArr[x][y])
        else:
            if newTempArr[x][y] >= Tnext:
                list2.append(newTempArr[x][y])

newMean = math.ceil(np.mean(list1))
newMean1 = math.ceil(np.mean(list2))

Tnext = (newMean + newMean1) / 2

oldMean = newMean
oldMean1 = newMean1
Tprev = Tnext
count = 0
while True:
    list1 = []
    list2 = []
    for x in range(0, 256, 1):
        for y in range(0, 256, 1):
            if newTempArr[x][y] < Tnext:
                list1.append(newTempArr[x][y])
            else:
                if newTempArr[x][y] >= Tnext:
                    list2.append(newTempArr[x][y])
    newMean = math.ceil(np.mean(list1))
    newMean1 = math.ceil(np.mean(list2))
    Tprev = Tnext
    Tnext = (newMean + newMean1) / 2

    count = count + 1

    if Tnext == Tprev:
        break


for x in range(0, 256, 1):
    for y in range(0, 256, 1):
        if newTempArr[x][y] < math.ceil(Tnext):
            newTempArr[x][y] = 0
        elif newTempArr[x][y] >= math.ceil(Tnext):
            newTempArr[x][y] = 255

#plt.imshow(newTempArr, cmap='gray')
#plt.show()

opsubImage4 = np.copy(newTempArr)

# consolidate the sub-images

outputImage = np.zeros((512,512),dtype='uint8')

#copy subimage 1
for x in range(0,256,1):
    for y in range(0,256,1):
        outputImage[x][y] = opsubImage1[x][y]

#copy subimage 2
x = 0
y = 0

for i in range(256,512,1):
    for j in range(0,256,1):
        outputImage[i][j] = opsubImage2[x][y]
        y = y + 1
    x = x + 1
    y = 0

#copy subimage 3
x = 0
y = 0

for i in range(0,256,1):
    for j in range(256,512,1):
        outputImage[i][j] = opsubImage3[x][y]
        y = y + 1
    x = x + 1
    y = 0

#copy subimage 4
x = 0
y = 0

for i in range(256,512,1):
    for j in range(256,512,1):
        outputImage[i][j] = opsubImage4[x][y]
        y = y + 1
    x = x + 1
    y = 0

adaptiveOutputArr = np.copy(outputImage)

####### 4. DUAL THRESHOLDING #########

Output512 = np.copy(tempArr)

#generate histogram for the respective image

histdata = [0] * 256

for x in range(0,512,1):
    for y in range(0,512,1):
        histdata[Output512[x][y]] = histdata[Output512[x][y]] + 1


firstHALF = [0] * 256
secondHALF = [0] * 256
for x in range(0,256,1):
    if x >= 0 and x <= 127:
        firstHALF[x] = histdata[x]
    else:
        secondHALF[x] = histdata[x]

max2half = np.max(secondHALF)

T1 = math.ceil(np.mean(Output512))
T2 = histdata.index(max2half)

#patition the image into three regions
#   R1  -   < T1
#   R2  -   T1<= x <= T2
#   R3  -   > T2

region1 = np.zeros((512,512),dtype='uint8')
region2 = np.zeros((512,512),dtype='uint8')
region3 = np.zeros((512,512),dtype='uint8')


for x in range(0,512,1):
    for y in range(0,512,1):
        if Output512[x][y] < T1:
            region1[x][y] = Output512[x][y]
            if Output512[x][y] == 0 : region1[x][y] = 1
        else:
            if Output512[x][y] >= T1 and Output512[x][y] <= T2:
                region2[x][y] = Output512[x][y]
            else:
                if Output512[x][y] > T2:
                    region3[x][y] = Output512[x][y]

# visit each pixel in R2 and if pixel has 4- nieghborhood belonging to R1 then assign pixel to R1

for x in range(0,512,1):
    for y in range(0,512,1):
        if region2[x][y] != 0 :
            if (x-1)>= 0 and (y-1) >=0 and (x+1) <= 511 and (y+1) <=511:
                if (region1[x - 1][y] != 0) and (region1[x][y + 1] != 0) and (region1[x][y - 1] != 0) and (
                    region1[x][y + 1] != 0) :
                    region2[x][y] = 0


# all the pixels from R2 which are not in R1 should be assigned to R3
for x in range(0,512,1):
    for y in range(0,512,1):
        if region2[x][y] != 0 :
            region3[x][y] = region2[x][y]
            region2[x][y] = 0

for x in range(0,512,1):
    for y in range(0,512,1):
        if region1[x][y] !=0 :
            region1[x][y] = 255
        else:
            region1[x][y] = 0

dualTOutputArr = np.copy(region1)

#display all four togehther

w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 2

fig.add_subplot(rows, columns, 1)
plt.title('Peakiness Detection')
plt.imshow(peakinessFinalImage, cmap='gray')

fig.add_subplot(rows, columns, 2)
plt.title('Iterative Thresholding')
plt.imshow(iterativeOutputArr, cmap='gray')

fig.add_subplot(rows, columns, 3)
plt.title('Adaptive Thresholding')
plt.imshow(adaptiveOutputArr, cmap='gray')

fig.add_subplot(rows, columns, 4)
plt.title('Dual Thresholding')
plt.imshow(dualTOutputArr, cmap='gray')

plt.show()