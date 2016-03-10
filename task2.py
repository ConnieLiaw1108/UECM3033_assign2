import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('cute.png')
[r,g,b] = [img[:,:,i] for i in range(3)]

fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img)
ax2.imshow(r, cmap = 'Reds')
ax3.imshow(g, cmap = 'Greens')
ax4.imshow(b, cmap = 'Blues')
plt.show()

#extract r, g and b into USV
rU, rS, rV = np.linalg.svd(r, full_matrices = False)
gU, gS, gV = np.linalg.svd(g, full_matrices = False)
bU, bS, bV = np.linalg.svd(b, full_matrices = False)

#calculate non-zero element in matrix S of r, g and b
red = np.count_nonzero(rS)
green = np.count_nonzero(gS)
blue = np.count_nonzero(bS)

#display the number of non-zero element of r, g and b in matrix S
print("Total number of non-zero elements in S of r:", red)
print("Total number of non-zero elements in S of g:", green)
print("Total number of non-zero elements in S of b:", blue)

#create 3 different matrices
rs1 = [0 for x in range (0,red)]
gs1 = [0 for x in range (0,green)]
bs1 = [0 for x in range (0,blue)]

#change the matrix S of r, g and b
#where only keep the first 30 non-zero elements
for i in range (0, red):
    if (i < 30):
        rs1[i] = rS[i]
    else:
        rs1[i] = 0
       
for i in range (0, green):
    if (i < 30):
        gs1[i] = gS[i]
    else:
        gs1[i] = 0

for i in range (0, blue):
    if (i < 30):
        bs1[i] = bS[i]
    else:
        bs1[i] = 0

#convert matrix s1 into diagonal matrix S1 of r, g and b
rS1 = np.diag(rs1)
gS1 = np.diag(gs1)
bS1 = np.diag(bs1)

#create new matrices by multiply matrix U, S1 and V of r, g, b
newR = np.dot(rU, np.dot(rS1,rV))
newG = np.dot(gU, np.dot(gS1,gV))
newB = np.dot(bU, np.dot(bS1,bV))

#create image from new matrices, save and display
badCute = np.dstack([newR, newG, newB])
plt.imsave('badCute.png', badCute)
plt.imshow(badCute)
plt.show()

#below same with the scenario above
#but just keep the first 200 non-zero elements instead of 30
rs2 = [0 for x in range (0,red)]
gs2 = [0 for x in range (0,green)]
bs2 = [0 for x in range (0,blue)]

for i in range (0, red):
    if (i < 200):
        rs2[i] = rS[i]
    else:
        rs2[i] = 0
        
for i in range (0, green):
    if (i < 200):
        gs2[i] = gS[i]
    else:
        gs2[i] = 0

for i in range (0, blue):
    if (i < 200):
        bs2[i] = bS[i]
    else:
        bs2[i] = 0        

rS2 = np.diag(rs2)
gS2 = np.diag(gs2)
bS2 = np.diag(bs2)

newR1 = np.dot(rU, np.dot(rS2,rV))
newG1 = np.dot(gU, np.dot(gS2,gV))
newB1 = np.dot(bU, np.dot(bS2,bV))

goodCute = np.dstack([newR1, newG1, newB1])
plt.imsave('goodCute', goodCute)
plt.imshow(goodCute)
plt.show()