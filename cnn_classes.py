import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

class Utils:
    def __init__(self):
        pass
    
    def iter_image_regions_conv(self, image, kernel_dim):
        k_w,k_h = kernel_dim, kernel_dim
        i_w,i_h = image.shape[1], image.shape[0]
        for i in range(i_h-k_h+1):
            for j in range(i_w-k_w+1):
                yield image[i:i+k_h, j:j+k_w], i, j

    def convolve(self, image, kernels, kernel_dim):
        imc = image.copy()
        # assume squares for kernel and input
        num_kernels = len(kernels)
        k_dim = kernel_dim
        output_img_dim = len(image[0])-k_dim+1

        output = np.zeros((num_kernels,output_img_dim,output_img_dim))

        for region, i, j in self.iter_image_regions_conv(imc, k_dim):
            conv = np.sum(region*kernels, axis=(1,2))
            for f in range(num_kernels):
                output[f,i,j] = conv[f]

        return output
    
    def convolve_color(self, image, kernels, kernel_dim):
        imc = image.copy()
        # assume squares for kernel and input
        num_kernels = len(kernels)
        k_dim = kernel_dim
        output_img_dim = len(image[0])-k_dim+1

        output = np.zeros((num_kernels,output_img_dim,output_img_dim))

        # image in shape of (x,y,channels)
        # kernels in shape (8, 3, k_dim, k_dim)
        for c in range(3):
            for region, i, j in self.iter_image_regions_conv(imc[:,:,c], k_dim):
                conv = np.sum(region*kernels[:,c,:,:], axis=(1,2))
                for f in range(num_kernels):
                    output[f,i,j] += (1/3)*conv[f]
            
        return output
    
    def iter_image_regions_pool(self, image, pool_dim):
        # pool dim will be 2 for this project
        p_w,p_h = pool_dim, pool_dim
        i_w,i_h = image.shape[1], image.shape[0]
        for i in range(i_h//p_h):
            for j in range(i_w//p_w):
                yield image[p_h*i:p_h*i+p_h, p_w*j:p_w*j+p_w], i, j

    def max_pool(self, channels, pool_dim):
        n, img_h, img_w = channels.shape
        output = np.zeros((n, img_h//pool_dim, img_w//pool_dim))
        for c in range(n):
            for region, i, j in self.iter_image_regions_pool(channels[c], pool_dim):
                max_reg = np.max(region)
                output[c, i, j] = max_reg
        return output
    
    
    def convolve_backprop(self, image, dOut, kernel_dim):
        d,h,w = dOut.shape
        dKernels = np.zeros((d, kernel_dim, kernel_dim))
        for img_region, i, j in self.iter_image_regions_conv(image, kernel_dim):
            for f in range(d):
                dKernels[f,:,:] += dOut[f, i, j]*img_region
        return dKernels
    
    def convolve_backprop_color(self, image, dOut, kernel_dim):
        d,h,w = dOut.shape
        dKernels = np.zeros((d, 3, kernel_dim, kernel_dim))
        for c in range(3):
            for img_region, i, j in self.iter_image_regions_conv(image[:,:,c], kernel_dim):
                for f in range(d):
                    dKernels[f,c,:,:] += (1/3)*dOut[f, i, j]*img_region
        return dKernels
    
    def max_pool_backprop(self, channels, dInput, pool_dim):
        d,h,w = channels.shape
        dPool = np.zeros((d, h, w))
        for c in range(d):
            for region, i, j in self.iter_image_regions_pool(channels[:,:,c], pool_dim):
                # we get the original 2 by 2 region that produced
                max_val = np.max(region)
                inds = np.argwhere(region == max_val)
                dPool[c, 2*i+inds[:,0], 2*j+inds[:,1]] = dInput[c,i,j]
        return dPool

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def loss(self, y,y_pred):
        epsilon = 0.0001
        log_loss = y*np.log2(y_pred+epsilon) + (1-y)*np.log2(1-y_pred+epsilon)
        return -log_loss
    

class CustomCNNGrayScale:
    def __init__(self, epochs=50, lr=0.01):
        self.EPOCHS = epochs
        self.filters = np.random.rand(8,3,3)/9 
        self.w2_dim = 2048 
        self.w1_dim = 256
        self.W2 = np.random.rand(self.w1_dim, self.w2_dim).astype('float64')*np.sqrt(2/(self.w2_dim))
        self.W1 = np.random.rand(1, self.w1_dim).astype('float64')*np.sqrt(2/(self.w1_dim))
        self.LR = lr
        self.utils = Utils()
    
    def fit(self, X_train, y_train):
        for i in range(self.EPOCHS):
            avg_loss = []
            for img, y in zip(X_train, y_train):
                
                # =============== FORWARD-PROPIGATION ====================

                # CONVOLUTIONAL LAYER
                output = self.utils.convolve(img, self.filters, 3)
                CONV_OUTPUT = output.copy()

                output = self.utils.max_pool(output, 2)
                MAXPOOL_OUTPUT = output.copy()

                # LINEAR LAYER
                linear_input = output.flatten().reshape((1,-1))
                linear_output = linear_input.dot(self.W2.T)
                W2_OUT = linear_output.copy()

                linear_output = self.utils.sigmoid(self.W1.dot(linear_output.T))

                # =============== PREDICTION ====================
                
                pred = linear_output
                avg_loss.append(self.utils.loss(y, pred))

                # =============== BACK-PROPIGATION ====================

                # LINEAR LAYER
                dW1 = (pred-y).dot(W2_OUT)
                dW2 = ((pred-y).T.dot(self.W1)).T.dot(linear_input)
                dInput = ((pred-y).T.dot(self.W1)).dot(self.W2)

                self.W1 -= self.LR*dW1
                self.W2 -= self.LR*dW2

                # CONVOLUTIONAL LAYER
                IMAGE = img.copy()

                dInput_r = dInput.copy().reshape(MAXPOOL_OUTPUT.shape)
                dOut = self.utils.max_pool_backprop(CONV_OUTPUT, dInput_r, 2)
                dFilters = self.utils.convolve_backprop(IMAGE, dOut, 3)

                self.filters -= self.LR*dFilters
                
            avg_loss = np.array(avg_loss)
            print(f'EPOCH:{i} - Avg NLLoss: {np.sum(avg_loss)}')
            
    def predict(self, X):
        ''' 
        Return a probability that the individual is wearing a mask [0-1].
        The method can only accept one image (grayscale at a time)
        '''
        output = self.utils.convolve(X, self.filters, 3)
        CONV_OUTPUT = output.copy()

        output = self.utils.max_pool(output, 2)
        MAXPOOL_OUTPUT = output.copy()

        # LINEAR LAYER
        linear_input = output.flatten().reshape((1,-1))
        linear_output = linear_input.dot(self.W2.T)
        W2_OUT = linear_output.copy()

        linear_output = self.utils.sigmoid(self.W1.dot(linear_output.T))
        
        return linear_output

class CustomCNNColor:
    def __init__(self, epochs=50, lr=0.01):
        self.EPOCHS = epochs
        self.filters = np.random.rand(8,3,3,3)
        self.w2_dim = 2048 
        self.w1_dim = 256
        self.W2 = np.random.rand(self.w1_dim, self.w2_dim).astype('float64')*np.sqrt(2/(self.w2_dim))
        self.W1 = np.random.rand(1, self.w1_dim).astype('float64')*np.sqrt(2/(self.w1_dim))
        self.LR = lr
        self.utils = Utils()
    
    def fit(self, X_train, y_train):
        for i in range(self.EPOCHS):
            avg_loss = []
            for img, y in zip(X_train, y_train):
                
                # =============== FORWARD-PROPIGATION ====================

                # CONVOLUTIONAL LAYER
                output = self.utils.convolve_color(img, self.filters, 3)
                CONV_OUTPUT = output.copy()

                output = self.utils.max_pool(output, 2)
                MAXPOOL_OUTPUT = output.copy()

                # LINEAR LAYER
                linear_input = output.flatten().reshape((1,-1))
                linear_output = linear_input.dot(self.W2.T)
                W2_OUT = linear_output.copy()

                linear_output = self.utils.sigmoid(self.W1.dot(linear_output.T))

                # =============== PREDICTION ====================
                
                pred = linear_output
                avg_loss.append(self.loss(y, pred))

                # =============== BACK-PROPIGATION ====================

                # LINEAR LAYER
                dW1 = (pred-y).dot(W2_OUT)
                dW2 = ((pred-y).T.dot(self.W1)).T.dot(linear_input)
                dInput = ((pred-y).T.dot(self.W1)).dot(self.W2)

                self.W1 -= self.LR*dW1
                self.W2 -= self.LR*dW2

                # CONVOLUTIONAL LAYER
                IMAGE = img.copy()

                dInput_r = dInput.copy().reshape(MAXPOOL_OUTPUT.shape)
                dOut = self.utils.max_pool_backprop(CONV_OUTPUT, dInput_r, 2)
                dFilters = self.utils.convolve_backprop_color(IMAGE, dOut, 3)

                self.filters -= self.LR*dFilters
                
            avg_loss = np.array(avg_loss)
            print(f'EPOCH:{i} - Avg NLLoss: {np.sum(avg_loss)}')
            
    def predict(self, X):
        ''' 
        Return a probability that the individual is wearing a mask [0-1].
        The method can only accept one image (grayscale at a time)
        '''
        output = self.utils.convolve_color(X, self.filters, 3)
        CONV_OUTPUT = output.copy()

        output = self.utils.max_pool(output, 2)
        MAXPOOL_OUTPUT = output.copy()

        # LINEAR LAYER
        linear_input = output.flatten().reshape((1,-1))
        linear_output = linear_input.dot(self.W2.T)
        W2_OUT = linear_output.copy()

        linear_output = self.utils.sigmoid(self.W1.dot(linear_output.T))
        
        return linear_output