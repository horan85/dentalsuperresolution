import tensorflow as tf
import numpy as np
import os
import cv2
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os


#use the first GPU, change CUDA_VISIBLE_DEVICES to a list to utilize more GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

SummaryDir=os.path.join(os.getcwd(), 'summaries/u_net_nobgremov')
BatchNum=16
InputDimension=[400,400,1]
GtDimensions=[400,400,1]
LearningRate=1e-4
NumIteration=1e6
NumTrainImages=5680



InputData = tf.placeholder(tf.float32, [BatchNum]+InputDimension) #network input
InputGT = tf.placeholder(tf.float32, [BatchNum]+GtDimensions) #network input

def LeakyReLU(Input):
	#leaky ReLU
	alpha=0.01
	Output =tf.maximum(alpha*Input,Input)
	return Output

def ConvBlock(Input,NumConv,KernelSize,LayerNum):
	#first convolution to scale the kernels
	with tf.variable_scope('conv'+str(LayerNum)):
		W = tf.get_variable('W',KernelSize)

		Input= tf.nn.conv2d(Input,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
	
		Bias  = tf.get_variable('B',[KernelSize[3]])
		Input= tf.add(Input, Bias)
		Input= LeakyReLU(Input)
		LayerNum+=1

	KernelSize[2]=KernelSize[3]
	for i in range(NumConv-1):
		with tf.variable_scope('conv'+str(LayerNum)):
			W = tf.get_variable('W',KernelSize)	
			Input= tf.nn.conv2d(Input,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
	
			beta= tf.get_variable('beta',[KernelSize[3]],initializer=tf.constant_initializer(0.0))
      			gamma=tf.get_variable('gamma',[KernelSize[3]],initializer=tf.constant_initializer(1.0))
      			Mean,Variance = tf.nn.moments(ConvResult,[0,1,2])
      			Input= tf.nn.batch_normalization(Input,Mean, Variance, beta, gamma,1e-10)
      
			#no bias is needed with batch norm
			#Bias  = tf.get_variable('B',[KernelSize[3]])
			#Input= tf.add(Input, Bias)

			Input= LeakyReLU(Input)
			LayerNum+=1
	return Input,LayerNum

CurrentInput =InputData

NumKernels=[32,64,128,256,512]
LayerNum=0
#left side of U
#first conv layer and #downsample
LayerLeft1,LayerNum = ConvBlock(CurrentInput ,1,[3,3,InputDimension[2],NumKernels[0]],LayerNum)
CurrentInput = tf.nn.max_pool(LayerLeft1 , ksize=[1,2, 2, 1], strides=[ 1, 2, 2, 1], padding='VALID')
#second conv layer
LayerLeft2,LayerNum = ConvBlock(CurrentInput ,2,[3,3,NumKernels[0],NumKernels[1]],LayerNum)
CurrentInput = tf.nn.max_pool(LayerLeft2 , ksize=[1,2, 2, 1], strides=[ 1, 2, 2, 1], padding='VALID')
#third conv layer
LayerLeft3,LayerNum = ConvBlock(CurrentInput ,2,[3,3,NumKernels[1],NumKernels[2]],LayerNum)
CurrentInput = tf.nn.max_pool(LayerLeft3 , ksize=[1,2, 2, 1], strides=[ 1, 2, 2, 1], padding='VALID')
#fourth conv layer
LayerLeft4,LayerNum = ConvBlock(CurrentInput ,2,[3,3,NumKernels[2],NumKernels[3]],LayerNum)
CurrentInput = tf.nn.max_pool(LayerLeft4 , ksize=[1,2, 2, 1], strides=[ 1, 2, 2, 1], padding='VALID')


CurrentInput,LayerNum = ConvBlock(CurrentInput ,2,[3,3,NumKernels[3],NumKernels[4]],LayerNum)

#right side of U
#upscale and convolution
with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[3], NumKernels[4]])
	LayerRight4=tf.nn.conv2d_transpose(CurrentInput, W,  [BatchNum, 50, 50, NumKernels[3]], [1, 2, 2, 1], padding='SAME', name=None)
	Bias  = tf.get_variable('B',[NumKernels[3]])
	LayerRight4=tf.add(LayerRight4,Bias )
	LayerRight4= LeakyReLU(LayerRight4)
	LayerNum +=1
CurrentInput=tf.concat([LayerRight4,LayerLeft4],3)

LayerRight4,LayerNum= ConvBlock(CurrentInput ,2,[3,3,NumKernels[4],NumKernels[3]],LayerNum)
with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[2], NumKernels[3]])
	LayerRight3=tf.nn.conv2d_transpose(LayerRight4, W,  [BatchNum, 100, 100, NumKernels[2]], [1, 2, 2 , 1], padding='SAME', name=None)
	Bias  = tf.get_variable('B',[NumKernels[2]])
	LayerRight3=tf.add(LayerRight3,Bias )
	LayerRight3= LeakyReLU(LayerRight3)
	LayerNum +=1
CurrentInput=tf.concat([LayerRight3,LayerLeft3 ],3)

LayerRight3,LayerNum= ConvBlock(CurrentInput ,2,[3,3,NumKernels[3],NumKernels[2]],LayerNum)
with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[1], NumKernels[2]])
	LayerRight2=tf.nn.conv2d_transpose(LayerRight3, W,  [BatchNum, 200, 200, NumKernels[1]], [1, 2, 2, 1], padding='SAME', name=None)
	Bias  = tf.get_variable('B',[NumKernels[1]])
	LayerRight2=tf.add(LayerRight2,Bias )
	LayerRight2= LeakyReLU(LayerRight2)
	LayerNum +=1
CurrentInput=tf.concat([LayerRight2,LayerLeft2 ],3)

LayerRight2,LayerNum= ConvBlock(CurrentInput ,2,[3,3,NumKernels[2],NumKernels[1]],LayerNum)
with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[0], NumKernels[1]])
	LayerRight1=tf.nn.conv2d_transpose(LayerRight2, W,  [BatchNum, 400, 400, NumKernels[0]], [1, 2, 2, 1], padding='SAME', name=None)
	Bias  = tf.get_variable('B',[NumKernels[0]])
	LayerRight1=tf.add(LayerRight1,Bias )
	LayerRight1= LeakyReLU(LayerRight1)
	LayerNum +=1
CurrentInput=tf.concat([LayerRight1,LayerLeft1 ],3)

LayerOut,LayerNum= ConvBlock(CurrentInput ,2,[3,3,NumKernels[1],NumKernels[0]],LayerNum)
with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[0], 1])
	LayerOut= tf.nn.conv2d(LayerOut,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
	Bias  = tf.get_variable('B',[1])
	LayerOut= tf.add(LayerOut, Bias)
#no nonlinearity at the end
#LayerOut= LeakyReLU(LayerOut)
Enhanced=LayerOut


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
 
    #average loss
    SoftL1Loss = tf.reduce_mean( SoftL1)
    L1Loss = tf.reduce_mean( AbsDif)
    L2Loss = tf.reduce_mean( tf.square(tf.subtract(InputGT,Enhanced)))



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
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)

#create image summary from the first 10 images
tf.summary.image('images', DataImages[1:10,:,:,:],  max_outputs=50)

#create scalar summaries for lsos and accuracy
tf.summary.scalar("SoftL1", SoftL1Loss)
tf.summary.scalar("L1", L1Loss )

SummaryOp = tf.summary.merge_all()

conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 1.0 #fraction of GPU used

Init = tf.global_variables_initializer()

with tf.Session(config=conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(SummaryDir,tf.get_default_graph())
	Saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    
	Step = 1
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
		if (Step%100)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Loss:" + str(L))
		#save samples
		if (Step%1000)==0:
			for i in range(5):
				temp = (OutputImages[i,:,:,0] - np.amin(OutputImages[i,:,:,0]))/(np.amax(OutputImages[i,:,:,0]) - np.amin(OutputImages[i,:,:,0]))*65535
				cv2.imwrite('samples_03/gt_'+str(Step).zfill(5)+'_'+str(TrainIndices[i]).zfill(5)+'.png',(Label[i,:,:,0]*65535).astype('uint16'))
				cv2.imwrite('samples_03/in_'+str(Step).zfill(5)+'_'+str(TrainIndices[i]).zfill(5)+'.png',(Data[i,:,:,0]*65535).astype('uint16'))
				cv2.imwrite('samples_03/out_'+str(Step).zfill(5)+'_'+str(TrainIndices[i]).zfill(5)+'.png',(temp).astype('uint16'))
			print('Saving model...')
			print(Saver.save(Sess, os.path.join(os.getcwd(), "checkpoint_u_03/")))
		SummaryWriter.add_summary(Summary,Step)
		Step+=1

