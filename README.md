# abcdef111
The code for Distortion_Convolution_Module_for_Semantic_Segmentation, IEEE TIM. Xing Hu, Yi An....

to run it , you need install pytorch, tqdm, and tensorboardX package.

to train and test, you can run python train.py --batch-size 8 --lr 0.007,and you will both get the training and testing results.

if you use the datasets of yourself, and the other dataset of panoramic image, you need update the intrinsics parameters of your camera in the file sperical.py in modeling/s2cnn.
