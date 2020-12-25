#!/usr/bin/env python
# coding: utf-8

# # Import the numpy package under the name np

# In[1]:


import numpy as np


# # Create a null vector of size 10

# In[3]:


a=np.zeros(10)


# In[4]:


a


# In[5]:


b=np.zeros((10,10))


# In[6]:


b


# # Create a vector with values ranging from 10 to 49

# In[10]:


c=np.arange(10,50)
c


# # Find the shape of previous array in question 3

# In[13]:


c.shape


# # Print the type of the previous array in question 3

# In[21]:


type(c)


# # Print the numpy version and the configuration

# In[32]:


np.__version__


# In[33]:


np.show_config()


# # Print the dimension of the array in question 3

# In[35]:


c.ndim


# # Create a boolean array with all the True values

# In[57]:


a=np.array([1,2,3,4,5],dtype=bool)
print(a)


# # Create a two dimensional array

# In[114]:


a=np.random.randn(3,3)
print(a)
print(a.ndim)


# # Create a three dimensional array

# In[116]:


a=np.random.randn(3,3,3)
print(a)
print(a.ndim)


# # Reverse a vector (first element becomes last)
# 

# In[75]:


a=np.arange(5)
print(a)
np.flip(a,axis=0)


# # Create a null vector of size 10 but the fifth value which is 1

# In[82]:


a=np.zeros(10)
print(a)


# In[84]:


a[4]=1
print(a)


# In[85]:


a[4]


# # Create a 3x3 identity matrix

# In[87]:


b=np.eye(3)


# In[88]:


b


# # arr = np.array([1, 2, 3, 4, 5])

# In[102]:


arr=np.array([1,2,3,4,5],dtype=float)
arr=arr.astype(float)
arr


# # arr1 = np.array([[1., 2., 3.],
# 
#             [4., 5., 6.]])  
# 
# arr2 = np.array([[0., 4., 1.],
# 
#            [7., 2., 12.]])
# Multiply arr1 with arr2

# In[103]:


arr1=np.array([[1., 2., 3.],
        [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],

       [7., 2., 12.]])
arr1*arr2


# # arr1 = np.array([[1., 2., 3.],
# 
#             [4., 5., 6.]]) 
# 
# arr2 = np.array([[0., 4., 1.],
# 
#             [7., 2., 12.]])
# Make an array by comparing both the arrays provided above

# In[111]:


arr1 = np.array([[1., 2., 3.],
        [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

        [7., 2., 12.]])
arr1>arr2


# In[112]:


arr1<arr2


# # Extract all odd numbers from arr with values(0-9)

# In[132]:


arr=np.arange(0,11)


# In[238]:


arr


# In[239]:


arr[1:10:2]


# # Replace all odd numbers to -1 from previous array

# In[135]:


a=np.arange(0,11)


# In[136]:


a


# In[142]:


np.where(a%2!=0,-1,a)


# # arr = np.arange(10)

# # Replace the values of indexes 5,6,7 and 8 to 12

# In[145]:


arr=np.arange(10)


# In[146]:


arr


# In[153]:


arr[5]=12
arr[6]=12
arr[7]=12
arr[8]=12
print(arr)


# # Create a 2d array with 1 on the border and 0 inside

# In[198]:


import numpy as np
x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)


# # arr2d = np.array([[1, 2, 3],
# 
#             [4, 5, 6], 
# 
#             [7, 8, 9]])
# 
# Replace the value 5 to 12

# In[6]:


import numpy as np
arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])

arr2d[1][1]=12
arr2d


# # arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# Convert all the values of 1st array to 64

# In[11]:


arr3d = np.array([[[1, 2, 3],[4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


# In[16]:


arr3d[0]=64
arr3d


# # Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[20]:


a=np.random.randint(0,9,size=(3,4))


# In[21]:


a


# In[29]:


a[0,0:]


# # Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
# 

# In[30]:


a=np.random.randint(0,9,size=(3,4))


# In[31]:


a


# In[35]:


a[1,1]


# # Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[36]:


a=np.random.randint(0,9,size=(3,4))


# In[41]:


a


# In[49]:


a[:2,2]


# # Create a 10x10 array with random values and find the minimum and maximum values

# In[94]:


a=np.random.randn(10)


# In[95]:


a.max()


# In[96]:


a.min()


# In[ ]:





# # 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# Find the common items between a and b

# In[63]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[65]:


print(np.intersect1d(a,b))


# # a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# Find the positions where elements of a and b match

# In[97]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[100]:


c=np.where(a==b)
print(c)


# # names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)

# # Find all the values from array data where the values from array names are not equal to Will

# In[101]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)


# In[106]:


print(data[~(names=="Will")])


# # names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# Find all the values from array data where the values from array names are not equal to Will and Joe

# In[113]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
mask=(names=="Will")|(names=="Joe")
data[~mask]


# # Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15

# In[130]:


a=np.random.randint(1,15,size=(5,3),dtype=int)


# In[131]:


a


# # Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[141]:


arr=np.random.randint(1,16,size=(2,2,4),dtype=int)


# In[142]:


arr


# # Swap axes of the array you created in Question 32

# In[147]:


arr=np.swapaxes(arr,1,2)


# In[148]:


arr


# # Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[169]:


a=np.random.random(10)
result=np.where(np.sqrt(a)<0.5,0,a)
print(a,result)


# # Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[237]:


a=np.random.random(12)
b=np.random.random(12)
c=np.maximum(a,b)
print(c)


# # names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# Find the unique names and sort them out!

# In[197]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names=np.unique(names)
names.sort()
print(names)


# # a = np.array([1,2,3,4,5]) b = np.array([5,6,7,8,9])
# From array a remove all items present in array b

# In[198]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
print(result)


# # Following is the input NumPy array delete column two and insert following new column in its place.
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])
# 
# newColumn = numpy.array([[10,10,10]])

# In[212]:


import numpy as numpy
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])

newColumn = numpy.array([[10,10,10]])


# In[223]:


a=np.delete(sampleArray,2,axis=1)
print(a)


# In[224]:


np.insert(a,2,newColumn,axis=1)


# # x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# Find the dot product of the above two matrix

# In[228]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
c=np.dot(x,y)
print(c)


# # Generate a matrix of 20 random values and find its cumulative sum

# In[235]:


a=np.random.randn(5,4)
np.cumsum(a)


# In[ ]:





# In[ ]:





# In[ ]:




