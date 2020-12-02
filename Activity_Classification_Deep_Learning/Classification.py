import os
import random
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal

import keras
from keras.layers.core import Masking
from keras.models import Model
from keras.layers import Dense,Activation,Input,Bidirectional,concatenate,Masking
from keras.layers.recurrent import SimpleRNN,LSTM
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# initialization.
def init():
	print('Init...')

	# input and output directories for reading and writing files
	datasetdir = 'datasets'
	outputdir = 'outputs'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	
	# experimental setting
	dataset_name = 'sbu' # 'tst', 'utkinect', 'sbu'
	validation_method = '2fold' # 'loso':Leave One Out, '2fold':2 Fold
	
	# parameters
	downsampling_ratio = 1
	resampling_len = 100
	n_epochs = 15
	s_batch = 20
	
	return datasetdir, outputdir, dataset_name, validation_method, downsampling_ratio, resampling_len, n_epochs, s_batch

# resample the signals
def resample(data,n_samples,action_lengths,method):
	print('Resampling...')
	
	if method=='slice':
		data = data[:,:,:,:,:,:,::downsampling_ratio]
	else:
		# not edited
		temp_data = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],resampling_len))
		for subject_idx in range(0,data.shape[0]):
			for action_idx in range(0,data.shape[2]):
				for trial_idx in range(0,n_samples[subject_idx,action_idx]):	
					l = action_lengths[subject_idx,trial_idx,action_idx]
					for subsignal_idx in range(0,data.shape[3]):
						if method=='downsample':
							s = data[subject_idx,trial_idx,action_idx,subsignal_idx,:]
						elif method=='resample':
							s = data[subject_idx,trial_idx,action_idx,subsignal_idx,0:l-1]
						temp_data[subject_idx,trial_idx,action_idx,subsignal_idx,:] = signal.resample(s,resampling_len)
		data = temp_data

	return data

# load the dataset
def load_dataset():
	print('Loading Dataset...')
	
	# load the dataset from file
	if dataset_name=='tst':
		dataset_path = os.path.join(datasetdir, 'TST.mat')
	elif dataset_name=='utkinect':
		dataset_path = os.path.join(datasetdir, 'UTKinect.mat')
	elif dataset_name=='sbu':
		dataset_path = os.path.join(datasetdir, 'SBU.mat')
	dataset = scipy.io.loadmat(dataset_path);
	data = dataset['skeleton']
	n_samples = dataset['number_of_samples']
	n_bodies = dataset['number_of_bodies']
	action_lengths = dataset['action_length']
	
	# permute the data to (subject,action,trial,body,joint,dimension,frame)
	data = np.transpose(data, (0,1,2,3,5,6,4))
	
	# resample the signals
	data = resample(data,n_samples,action_lengths,'slice')
	
	return data, n_samples, n_bodies, action_lengths

# partitioning the dataset to training and testing
def partition(training_idx,testing_idx):
	
	# print the testing subject indices
	print('\ttesting on: ',testing_idx)
			
	# create an array containing the action labels
	action_labels = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
	for subject_idx in range(0,data.shape[0]):
		for action_idx in range(0,data.shape[1]):
			for trial_idx in range(0,data.shape[2]):
				action_labels[subject_idx,action_idx,trial_idx] = int(action_idx)
	
	# seperate the data to train and test by subjects
	training_data = data[training_idx,:,:,:,:,:,:]
	training_n_samples = n_samples[training_idx,:]
	training_n_bodies = n_bodies[training_idx,:,:]
	training_lengths = action_lengths[training_idx,:,:]
	training_labels = action_labels[training_idx,:,:]
	testing_data = data[testing_idx,:,:,:,:,:,:]
	testing_n_samples = n_samples[testing_idx,:]
	testing_n_bodies = n_bodies[testing_idx,:,:]
	testing_lengths = action_lengths[testing_idx,:,:]
	testing_labels = action_labels[testing_idx,:,:]
	
	# dimensions with size 1 have been automatically removed. add the removed dimensions if needed
	if training_data.ndim==6:
		training_data = np.expand_dims(training_data,axis=0)
		training_n_samples = np.expand_dims(training_n_samples,axis=0)
		training_n_bodies = np.expand_dims(training_n_bodies,axis=0)
		training_lengths = np.expand_dims(training_lengths,axis=0)
		training_labels = np.expand_dims(training_labels,axis=0)
	if testing_data.ndim==6:
		testing_data = np.expand_dims(testing_data,axis=0)
		testing_n_samples = np.expand_dims(testing_n_samples,axis=0)
		testing_n_bodies = np.expand_dims(testing_n_bodies,axis=0)
		testing_lengths = np.expand_dims(testing_lengths,axis=0)
		testing_labels = np.expand_dims(testing_labels,axis=0)
	
	# reshape the data to remove subject dependency
	training_data_reshaped = []
	training_n_bodies_reshaped = []
	training_lengths_reshaped = []
	training_labels_reshaped = []
	training_data_transposed = np.transpose(training_data, (3,4,5,6,0,1,2))
	for subject_idx in range(0,training_data.shape[0]):
		for action_idx in range(0,training_data.shape[1]):
			for trial_idx in range(0,training_n_samples[subject_idx,action_idx]):
				sample_data = training_data_transposed[:,:,:,:,subject_idx,action_idx,trial_idx]
				sample_n_bodies = training_n_bodies[subject_idx,action_idx,trial_idx]
				sample_length = training_lengths[subject_idx,action_idx,trial_idx]
				sample_label = training_labels[subject_idx,action_idx,trial_idx]
				training_data_reshaped.append(sample_data.tolist())
				training_n_bodies_reshaped.append(sample_n_bodies)
				training_lengths_reshaped.append(sample_length)
				training_labels_reshaped.append(sample_label)
	training_data_reshaped = np.array(training_data_reshaped)
	training_n_bodies_reshaped = np.array(training_n_bodies_reshaped)
	training_lengths_reshaped = np.array(training_lengths_reshaped)
	training_labels_reshaped = np.array(training_labels_reshaped)
	
	testing_data_reshaped = []
	testing_n_bodies_reshaped = []
	testing_lengths_reshaped = []
	testing_labels_reshaped = []
	testing_data_transposed = np.transpose(testing_data, (3,4,5,6,0,1,2))
	for subject_idx in range(0,testing_data.shape[0]):
		for action_idx in range(0,testing_data.shape[1]):
			for trial_idx in range(0,testing_n_samples[subject_idx,action_idx]):
				sample_data = testing_data_transposed[:,:,:,:,subject_idx,action_idx,trial_idx]
				sample_n_bodies = testing_n_bodies[subject_idx,action_idx,trial_idx]
				sample_length = testing_lengths[subject_idx,action_idx,trial_idx]
				sample_label = testing_labels[subject_idx,action_idx,trial_idx]
				testing_data_reshaped.append(sample_data.tolist())
				testing_n_bodies_reshaped.append(sample_n_bodies)
				testing_lengths_reshaped.append(sample_length)
				testing_labels_reshaped.append(sample_label)
	testing_data_reshaped = np.array(testing_data_reshaped)
	testing_n_bodies_reshaped = np.array(testing_n_bodies_reshaped)
	testing_lengths_reshaped = np.array(testing_lengths_reshaped)
	testing_labels_reshaped = np.array(testing_labels_reshaped)
	
	# concatenate the bodies (zero-pad)
	training_data_concatenated = np.zeros((training_data_reshaped.shape[0],training_data_reshaped.shape[1]*training_data_reshaped.shape[2],training_data_reshaped.shape[3],training_data_reshaped.shape[4]))
	for sample_idx in range(0,training_data_reshaped.shape[0]):
		for body_idx in range(0,training_n_bodies_reshaped[sample_idx]):
			for joint_idx in range(0,training_data_reshaped.shape[2]):
				training_data_concatenated[sample_idx,body_idx*training_data_reshaped.shape[2]+joint_idx,:,:] = training_data_reshaped[sample_idx,body_idx,joint_idx,:,:]
		
	testing_data_concatenated = np.zeros((testing_data_reshaped.shape[0],testing_data_reshaped.shape[1]*testing_data_reshaped.shape[2],testing_data_reshaped.shape[3],testing_data_reshaped.shape[4]))
	for sample_idx in range(0,testing_data_reshaped.shape[0]):
		for body_idx in range(0,testing_n_bodies_reshaped[sample_idx]):
			for joint_idx in range(0,testing_data_reshaped.shape[2]):
				testing_data_concatenated[sample_idx,body_idx*testing_data_reshaped.shape[2]+joint_idx,:,:] = testing_data_reshaped[sample_idx,body_idx,joint_idx,:,:]
	
	# reshape the data to form the subsignals
	training_data = np.reshape(training_data_concatenated,(training_data_concatenated.shape[0],training_data_concatenated.shape[1]*training_data_concatenated.shape[2],training_data_concatenated.shape[3]))
	testing_data = np.reshape(testing_data_concatenated,(testing_data_concatenated.shape[0],testing_data_concatenated.shape[1]*testing_data_concatenated.shape[2],testing_data_concatenated.shape[3]))
	
	# convert the data to keras compatible format
	n_classes = int(training_labels.max()+1)
	training_data = np.transpose(training_data, (0,2,1))
	training_lengths = training_lengths_reshaped
	training_labels = keras.utils.to_categorical(training_labels_reshaped, num_classes=n_classes)
	testing_data = np.transpose(testing_data, (0,2,1))
	testing_lengths = testing_lengths_reshaped
	testing_labels = keras.utils.to_categorical(testing_labels_reshaped, num_classes=n_classes)
	
	# # seperate the data parts
	# trunc_data = np.copy(data)
	# left_arm_data = np.copy(data)
	# right_arm_data = np.copy(data)
	# left_leg_data = np.copy(data)
	# right_leg_data = np.copy(data)
	# if dataset_name=='tst' or dataset_name=='utkinect':
		# trunc_data = trunc_data[:,:,:,:,0:4,:,:]
		# left_arm_data = left_arm_data[:,:,:,:,4:8,:,:]
		# right_arm_data = right_arm_data[:,:,:,:,8:12,:,:]
		# left_leg_data = left_leg_data[:,:,:,:,12:16,:,:]
		# right_leg_data = right_leg_data[:,:,:,:,16:20,:,:]
	# elif dataset_name=='sbu':
		# trunc_data = trunc_data[:,:,:,:,0:3,:,:]
		# left_arm_data = left_arm_data[:,:,:,:,3:6,:,:]
		# right_arm_data = right_arm_data[:,:,:,:,6:9,:,:]
		# left_leg_data = left_leg_data[:,:,:,:,9:12,:,:]
		# right_leg_data = right_leg_data[:,:,:,:,12:15,:,:]
	
	return training_data, training_lengths, training_labels, testing_data, testing_lengths, testing_labels, n_classes
		
# training the network
def train_model(training_left_arm_data, training_right_arm_data, training_trunc_data, training_left_leg_data, training_right_leg_data, n_classes):
	
	# defining layers
	left_arm_data_input = Input(shape=(None,training_left_arm_data.shape[2],))
	right_arm_data_input = Input(shape=(None,training_right_arm_data.shape[2],))
	trunc_data_input = Input(shape=(None,training_trunc_data.shape[2],))
	left_leg_data_input = Input(shape=(None,training_left_leg_data.shape[2],))
	right_leg_data_input = Input(shape=(None,training_right_leg_data.shape[2],))
		
	# masking layer
	masked_left_arm_data = Masking(0)(left_arm_data_input)
	masked_right_arm_data = Masking(0)(right_arm_data_input)
	masked_trunc_data = Masking(0)(trunc_data_input)
	masked_left_leg_data = Masking(0)(left_leg_data_input)
	masked_right_leg_data = Masking(0)(right_leg_data_input)
	
	left_arm_data_output = Bidirectional(SimpleRNN(12, return_sequences = True))(left_arm_data_input)
	right_arm_data_output = Bidirectional(SimpleRNN(12, return_sequences = True))(right_arm_data_input)
	trunc_data_output = Bidirectional(SimpleRNN(12, return_sequences = True))(trunc_data_input)
	left_leg_data_output = Bidirectional(SimpleRNN(12, return_sequences = True))(left_leg_data_input)
	right_leg_data_output = Bidirectional(SimpleRNN(12, return_sequences = True))(right_leg_data_input)
	
	left_right_arm_data = concatenate([left_arm_data_output, right_arm_data_output])
	right_arm_data_trunc_data = concatenate([right_arm_data_output, trunc_data_output ])
	left_leg_data_trunc_data = concatenate([left_leg_data_output, trunc_data_output])
	left_right_leg_data = concatenate([left_leg_data_output, right_leg_data_output])
	
	left_right_arm_data_output = Bidirectional(SimpleRNN(24, return_sequences=True))(left_right_arm_data)
	right_arm_data_trunc_data_output= Bidirectional(SimpleRNN(24, return_sequences=True))(right_arm_data_trunc_data)
	left_leg_data_trunc_data_output = Bidirectional(SimpleRNN(24, return_sequences=True))(left_leg_data_trunc_data)
	left_right_leg_data_output = Bidirectional(SimpleRNN(24, return_sequences=True))(left_right_leg_data)
	
	upper_data = concatenate([left_right_arm_data_output, right_arm_data_trunc_data_output])
	lower_data = concatenate([left_leg_data_trunc_data_output, left_right_leg_data_output])
	
	upper_data_output = Bidirectional(SimpleRNN(30, return_sequences=True))(upper_data)
	lower_data_output = Bidirectional(SimpleRNN(30, return_sequences=True))(lower_data)
	
	whole_data = concatenate([upper_data_output , lower_data_output])
	
	x = Bidirectional(LSTM(100, return_sequences=True))(whole_data)
	x = Bidirectional(LSTM(100, return_sequences=False))(x)
	x = Dense(n_classes)(x)
	
	output = Activation('softmax')(x)
	
	# defining model
	model = Model(inputs=[left_arm_data_input,right_arm_data_input,trunc_data_input,left_leg_data_input,right_leg_data_input], outputs=output)
	
	# compiling model
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model
	
def classification(training_data,training_lengths,training_labels,testing_data,testing_lengths,testing_labels,n_classes,tot_testing_labels,tot_predicted_labels):
	
	print(training_data.shape,training_lengths.shape,training_labels.shape)
	print(testing_data.shape,testing_lengths.shape,testing_labels.shape)
	
	# training model
	model = train_model(training_left_arm_data, training_right_arm_data, training_trunc_data, training_left_leg_data, training_right_leg_data, n_classes)
	model.fit([training_left_arm_data,training_right_arm_data,training_trunc_data,training_left_leg_data,training_right_leg_data], training_labels, epochs=n_epochs, batch_size=s_batch, verbose=0)

	# testing model
	predicted_proba = model.predict([testing_left_arm_data,testing_right_arm_data,testing_trunc_data,testing_left_leg_data,testing_right_leg_data], batch_size=1)
	
	# convert the labels to sklearn compatible format
	predicted_labels = np.argmax(predicted_proba,axis=1)
	testing_labels = np.argmax(testing_labels,axis=1)
	
	# print the recognition accuracy
	print('\taccuracy: {0:.3f}'.format(metrics.accuracy_score(testing_labels,predicted_labels)))
	
	# append the labels to the total labels
	tot_testing_labels.extend(testing_labels)
	tot_predicted_labels.extend(predicted_labels)
	
	return tot_testing_labels,tot_predicted_labels
	
# cross validation
def cross_validation():
	print('Cross Validation...')

	tot_testing_labels = []
	tot_predicted_labels = []
	n_subjects = data.shape[0]
	
	# leave one out
	if validation_method=='loso':
		indices = np.random.permutation(n_subjects)
		for subject_idx in range(0,n_subjects):
			training_idx = indices
			training_idx = np.delete(training_idx,subject_idx)
			testing_idx = indices[subject_idx]
			training_data, training_lengths, training_labels, testing_data, testing_lengths, testing_labels, n_classes = partition(training_idx,testing_idx)
			tot_testing_labels,tot_predicted_labels = classification(training_data,training_lengths,training_labels,testing_data,testing_lengths,testing_labels,n_classes,tot_testing_labels,tot_predicted_labels)
		
	# 2-fold
	elif validation_method=='2fold':
		K = 2
		indices = np.random.permutation(n_subjects)
		for pivot in range(0,K):
			if pivot<K-1:
				testing_idx = indices[int(pivot*math.ceil(n_subjects/K)):int((pivot+1)*math.ceil(n_subjects/K))]
				training_idx = np.copy(indices)
				training_idx = np.delete(training_idx, range(int(pivot*math.ceil(n_subjects/K)),int((pivot+1)*math.ceil(n_subjects/K))))
			else:
				testing_idx = indices[int(pivot*math.ceil(n_subjects/K)):]
				training_idx = indices[0:int(pivot*math.ceil(n_subjects/K))]
			training_data, training_lengths, training_labels, testing_data, testing_lengths, testing_labels, n_classes = partition(training_idx,testing_idx)
			tot_testing_labels,tot_predicted_labels = classification(training_data,training_lengths,training_labels,testing_data,testing_lengths,testing_labels,n_classes,tot_testing_labels,tot_predicted_labels)	
				
	return tot_testing_labels,tot_predicted_labels
	
# plot the confusion matrix
def plot_confusion_matrix(test_lbls, predicted_lbls):

	# class names
	if dataset_name=='tst':
		classes = ['sit','grasp','walk','lay','fall front','fall back','fall side','fall EndUpSit']
	elif dataset_name=='utkinect':
		classes = ['walk','sitDown','standUp','pickUp','carry','throw','push','pull','waveHands','clapHands']
	
	# calculate and normalize confusion matrix
	cnf_matrix = confusion_matrix(test_lbls, predicted_lbls)
	cnf_matrix = cnf_matrix.astype('int')
	norm_cnf_matrix = np.copy(cnf_matrix)
	norm_cnf_matrix = norm_cnf_matrix.astype('float')
	for row in range(0,cnf_matrix.shape[0]):
		s = cnf_matrix[row,:].sum()
		if s > 0:
			for col in range(0,cnf_matrix.shape[0]):
				norm_cnf_matrix[row,col] = np.double(cnf_matrix[row,col]) / s
	
	# print confusion matrix
	np.set_printoptions(precision=2)
	# print('\nConfusion Matrix=\n', cnf_matrix, '\n', '\nNormalized Confusion Matrix=\n', norm_cnf_matrix, '\n')
	
	# save confusion matrix as text
	np.savetxt(os.path.join(outputdir, 'Confusion Matrix.txt'), cnf_matrix, delimiter='\t', fmt='%d')
	
	# plot confusion matrix
	plt.figure()
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, int(cnf_matrix[i, j]), horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(outputdir, 'Confusion Matrix.jpg'), bbox_inches='tight', dpi=300)
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()
	
	# plot normalized confusion matrix
	plt.figure()
	plt.imshow(norm_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = norm_cnf_matrix.max() / 2.
	for i, j in itertools.product(range(norm_cnf_matrix.shape[0]), range(norm_cnf_matrix.shape[1])):
		plt.text(j, i, float("{0:.2f}".format(norm_cnf_matrix[i, j])), horizontalalignment="center", color="white" if norm_cnf_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(outputdir, 'Normalized Confusion Matrix.jpg'), bbox_inches='tight', dpi=600)
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()
	
	return

# calaculate the classification result
def result(test_labels, predicted_labels):
	
	# print the recognition accuracy
	print('Test accuracy: {0:.3f}'.format(metrics.accuracy_score(test_labels,predicted_labels)))
	
	# plot confusion matrix
	plot_confusion_matrix(test_labels, predicted_labels)
	
	return
	
# program main. main variables will be accesible within all functions without any need to passing.
if __name__ == '__main__':
	
	datasetdir, outputdir, dataset_name, validation_method, downsampling_ratio, resampling_len, n_epochs, s_batch = init()

	data, n_samples, n_bodies, action_lengths = load_dataset()
	
	tot_testing_labels,tot_predicted_labels = cross_validation()

	result(tot_testing_labels,tot_predicted_labels)
	