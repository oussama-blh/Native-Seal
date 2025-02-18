# 1) Time Taken for Encoding the First Convolution Layer Weights

## For Python Notebook  
**Time taken:** 250 milliseconds  

## Without Parallelization  
**Time taken:** 373 milliseconds  

## With OpenMP  
- Time taken for weights encoding: 82 milliseconds  
- Time taken for weights encoding: 77 milliseconds  
- Time taken for weights encoding: 87 milliseconds  
- Time taken for weights encoding: 85 milliseconds  
- Time taken for weights encoding: 64 milliseconds  
- Time taken for weights encoding: 60 milliseconds  
- Time taken for weights encoding: 52 milliseconds  
- Time taken for weights encoding: 46 milliseconds  
- Time taken for weights encoding: 40 milliseconds  
- Time taken for weights encoding: 35 milliseconds
- Time taken for weights encoding: 44 milliseconds

# 2) Time Taken for Image Encryption

## For Python Notebook  
**Time taken:** 6980 milliseconds  

### For c++ without Parallelization 
**Time taken for image encryption:** 12,365 milliseconds  

### For c++ with Parallelization 
- Time taken for image encryption: 2938 milliseconds  
- Time taken for image encryption: 1952 milliseconds  
- Time taken for image encryption: 1915 milliseconds  
- Time taken for image encryption: 1827 milliseconds  

# Time taken by the inference  :

## For the first convolutional layer :

### For Python Notebook  
**Time taken:** 67.79 seconds  

### For c++ without Parallelization 
**Time taken is:** approx 5 mins

### For c++ with Parallelization 
- Time taken for convolution: 89904 milliseconds  
- Time taken for convolution: 86001 milliseconds   
- Time taken for convolution: 90582 milliseconds  
- Time taken for convolution: 100901 milliseconds 
- Time taken for convolution: 95178 milliseconds

## For the Second convolutional layer :

### For Python Notebook  
**Time taken:** 206.54 seconds  
**Time taken:** 224.54 seconds 

### For c++ with Parallelization 
- Time taken for convolution: 78434 milliseconds   
- Time taken for convolution: 87054 milliseconds    
- Time taken for convolution: 81982 milliseconds  
- Time taken for convolution: 100901 milliseconds 
- Time taken for convolution: 95178 milliseconds
  
