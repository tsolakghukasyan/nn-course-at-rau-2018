## HW3. Compute the number of weights in a Keras CNN model

A simple parser of Keras CNN model descriptions that computes the total number of trainable params.

This implementation makes several assumptions regarding the input description:  
- in Conv2D layer, filter count is the first specified parameter   
- in Conv2D layer, kernel size is specified as named argument (i.e. 'kernel_size=...')  
- only Conv2D, Dense, MaxPooling2D, Droput, Flatten layers are accepted.  
