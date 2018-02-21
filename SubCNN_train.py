import tensorflow as tf
import numpy as np
import os
import cv2
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

#use the ifrst GPU, change CUDA_VISIBLE_DEVICES to a list to utilize more GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

SummaryDir=os.path.join(os.getcwd(), 'summaries/CNN_nobgremov')
BatchNum=64
InputDimension=[200,200,1]
GtDimensions=[400,400,1]
LearningRate=1e-4
NumIteration=100000
NumTrainImages=5680
NumKernels = [16,32,32,64,64,4]
FilterSizes =  [3,3,3,3,3,3]

InputData = tf.placeholder(tf.float32, [BatchNum]+InputDimension) #network input
InputGT = tf.placeholder(tf.float32, [BatchNum]+GtDimensions) #network input


def MakeConvNet(Input,Size):
	with tf.device('/gpu:0'):
		CurrentInput = InputData
		CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
		for i in range(len(NumKernels)): #number of layers
			with tf.variable_scope('conv'+str(i)):
				NumKernel=NumKernels[i]

				FilterSize = FilterSizes[i]
				W = tf.get_variable('W',[FilterSize,FilterSize,NumKernel,CurrentFilters])
				Bias = tf.get_variable('Bias',[NumKernel],initializer=tf.constant_initializer(0.1))
		
				CurrentFilters = NumKernel
				#ConvResult = tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
				ConvResult = tf.nn.conv2d_transpose(CurrentInput,W,[BatchNum,InputDimension[0],InputDimension[1],NumKernel],strides=[1,1,1,1])
				beta= tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))
      				gamma=tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))
      				Mean,Variance = tf.nn.moments(ConvResult,[0,1,2])
      				ConvResult= tf.nn.batch_normalization(ConvResult,Mean, Variance, beta, gamma,1e-10)
      
				#no bias is needed with batch norm
				#ConvResult= tf.add(ConvResult, Bias)
			
				#ReLU = tf.nn.relu(ConvResult)
				#leaky ReLU
				alpha=0.01
				ReLU=tf.maximum(alpha*ConvResult,ConvResult)
			
				#CurrentInput = tf.nn.max_pool(ReLU,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
				CurrentInput =ReLU
		Out=tf.depth_to_space(CurrentInput,GtDimensions[0]/InputDimension[0])   
    
		#Out=CurrentInput
	return Out

Enhanced=MakeConvNet(InputData,InputDimension)


# Define loss and optimizer
with tf.name_scope('loss'):
    # L1 loss
    AbsDif=tf.abs(tf.subtract(InputGT,Enhanced))
    
    #this part implements soft L1
    Comp = tf.constant(np.ones(AbsDif.shape), dtype = tf.float32)
    SmallerThanOne = tf.cast(tf.greater(Comp, AbsDif),tf.float32)
    LargerThanOne= tf.cast(tf.greater(AbsDif, Comp ),tf.float32)   
    ValuestoKeep=tf.subtract(AbsDif, tf.multiply(SmallerThanOne ,AbsDif))
    ValuestoSquare=tf.subtract(AbsDif, tf.multiply(LargerThanOne,AbsDif))
    SoftL1= tf.add(ValuestoKeep, tf.square(ValuestoSquare)) 
    
    # average loss
    SoftL1Loss = tf.reduce_mean( SoftL1)
    L1Loss = tf.reduce_mean( AbsDif)
    L2Loss = tf.reduce_mean( tf.square(tf.subtract(InputGT,Enhanced)))

InputDimension
with tf.name_scope('optimizer'):	
    #Use ADAM optimizer this is currently the best performing training algorithm in most cases
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(SoftL1Loss )


#preread images
OrigImages=np.zeros([NumTrainImages]+GtDimensions)
DataImages=np.zeros([NumTrainImages]+InputDimension)
Diffs=[]
GTPath='dental_registered/gt'
TrainPath='dental_registered/train'
Index=0
orig_min = 2198
orig_max = 23038
low_min = 31701
low_max = 64830
for file in os.listdir(GTPath):
    if os.path.isfile(os.path.join(GTPath, file)):
        if Index<NumTrainImages:
            OrigImg=cv2.imread(os.path.join(GTPath, file),-1)
            OrigImg[OrigImg < orig_min] = orig_min
            OrigImg[OrigImg > orig_max] = orig_max
            OrigImg = (OrigImg-orig_min)/(orig_max - orig_min)
            OrigImg=cv2.resize(OrigImg, (GtDimensions[0], GtDimensions[1]))
            LowImg=cv2.imread(os.path.join(TrainPath, file),-1)
            LowImg[LowImg < low_min] = low_min
            LowImg[LowImg > low_max] = low_max
            LowImg = (LowImg-low_min)/(low_max - low_min)
            LowImg=cv2.resize(LowImg, (InputDimension[0], InputDimension[1]))	          
                        
            #UpscaledImg=cv2.resize(LowImg, (GtDimensions[0], GtDimensions[1]))#/255.0
            #D=np.sum(np.square(OrigImg.astype(float)-UpscaledImg.astype(float)))
            #Diffs.append(D)
            #if D<1.8*1e7:
            OrigImages[Index,:,:,0]=OrigImg
            DataImages[Index,:,:,0]=LowImg
            Index+=1
print(Index)
#NumTrainImages=1029
#OrigImages=OrigImages[:1029,:,:,:]
#DataImages=DataImages[:1029,:,:,:]
#plt.plot(Diffs)
#plt.savefig('compare.png')


#create summaries

#histogram sumamries about the distribution of the variables
#for v in tf.trainable_variables():
#	tf.summary.histogram(v.name[:-2], v)

#create image summary from the first 10 images
tf.summary.image('images', DataImages[1:10,:,:,:],  max_outputs=50)

#create scalar summaries for lsos and accuracy
tf.summary.scalar("SoftL1", SoftL1Loss)
tf.summary.scalar("L1", L1Loss )

SummaryOp = tf.summary.merge_all()

conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.9 #fraction of GPU used

Init = tf.global_variables_initializer()

with tf.Session(config=conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(SummaryDir,tf.get_default_graph())
	Saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    
	Step = 100
	# Keep training until reach max iterations - other stopping criterion could be added
	while Step < NumIteration:
		#create batch 
		TrainIndices = random.sample(range(OrigImages.shape[0]), BatchNum)
		#TODO add data augmentation!!!
		Data=DataImages[TrainIndices,:,:,:]
		Label=OrigImages[TrainIndices,:,:,:]
		#execute teh session
		Summary,_,L,OutputImages = Sess.run([SummaryOp,Optimizer, SoftL1Loss ,Enhanced], feed_dict={InputData: Data, InputGT: Label})
		#print loss and accuracy at every 10th iteration
		if (Step%1000)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Loss:" + str(L))
		#save samples
		if (Step%1000)==0:
			for i in range(5):
				temp = (OutputImages[i,:,:,0] - np.amin(OutputImages[i,:,:,0]))/(np.amax(OutputImages[i,:,:,0]) - np.amin(OutputImages[i,:,:,0]))*65535
				cv2.imwrite('samples_01/gt_'+str(Step).zfill(5)+'_'+str(TrainIndices[i]).zfill(5)+'.png',(Label[i,:,:,0]*65535).astype('uint16'))
				cv2.imwrite('samples_01/in_'+str(Step).zfill(5)+'_'+str(TrainIndices[i]).zfill(5)+'.png',(Data[i,:,:,0]*65535).astype('uint16'))
				cv2.imwrite('samples_01/out_'+str(Step).zfill(5)+'_'+str(TrainIndices[i]).zfill(5)+'.png',(temp).astype('uint16'))
			print('Saving model...')
			#print(Saver.save(Sess, os.path.join(os.getcwd(), "checkpoint_01/model")),Step)
#	if (Step%10000)==0:
#		SummaryWriter.add_summary(Summary,Step)
			print(Saver.save(Sess, "./checkpoint_01/model",Step))
		Step+=1

