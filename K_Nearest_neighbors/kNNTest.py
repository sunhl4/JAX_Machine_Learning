'''
Created on Oct 27, 2010

@author: Peter
'''
from numpy import *
import matplotlib.pyplot as plt

from K_Nearest_neighbors import JAX_kNN

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat,datingLabels = JAX_kNN.File2Matrix('datingTestSet.txt')
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
#ax.axis([-2,25,-0.2,2.0])
ax.axis([-4000,100000,-2,25])


#plt.xlabel('Percentage of Time Spent Playing Video Games')
#plt.ylabel('Liters of Ice Cream Consumed Per Week')

plt.xlabel('Frequent flyer miles earned per year')
plt.ylabel('Percentage of Time Spent Playing Video Games')
plt.show()

#Test dating class
# kNN.datingClassTest()
