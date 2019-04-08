import numpy as np
import pickle
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.datasets import mnist

class EncodingCluster:

	def __init__(self,number_of_concepts,input_filename):
		
		self.number_of_clusters = number_of_concepts
		self.input_file = input_filename
		self.encodings = None
		self.encoding_cluster = None
		self.sample_shape = None
		self.index_map = None
		self.labels = None

	def read_data(self):
		
		file_obj = open(self.input_file,'rb')
		encodings = pickle.load(file_obj)

		for i in range(len(encodings)):
			encodings[i] = encodings[i][0][0]

		self.encodings = encodings
		self.sample_shape = encodings[0].shape

	def create_cluster(self):
		
		cluster = KMeans(n_clusters=self.number_of_clusters, random_state=0)

		self.encoding_cluster = cluster

	def map_indexes(self):
		self.labels = self.encoding_cluster.labels_

		idx_map = {}

		for i in range(len(self.labels)):
			label = self.labels[i]

			if(label not in idx_map):
				idx_map[label] = []

			idx_map[label].append(i)

		self.index_map = idx_map

	def fit(self):
		self.encoding_cluster = self.encoding_cluster.fit(self.encodings)
		self.map_indexes()

	def predict(self,sample,multiple = False):

		if(not multiple):
			if(sample.shape != self.sample_shape):
				print("Shape Error.")
				return

			sample = np.array(sample).reshape(1,-1)

		else:
			for i in sample:
				if(i.shape != self.sample_shape):
					print("Shape Error.")
					return

			sample = sample.reshape(-1,self.sample_shape)

		predictions = self.encoding_cluster.predict(sample)

		return predictions

	def return_cluster_centroids(self):
		return self.encoding_cluster.cluster_centers_

	def return_cluster_elements(self,label):
		return self.index_map[label]


class ThetaCluster:

	def __init__(self,number_of_concepts,number_of_classes,input_filename):
		
		self.number_of_clusters = number_of_concepts
		self.number_of_classes = number_of_classes
		self.input_file = input_filename
		self.theta_values = None
		self.theta_cluster = None
		self.sample_shape = None
		self.index_map = None
		self.labels = None

	def read_data(self):
		
		file_obj = open(self.input_file,'rb')
		theta_values = pickle.load(file_obj)

		for i in range(len(theta_values)):
			theta_values[i] = theta_values[i].reshape(self.number_of_clusters*self.number_of_classes,)

		self.theta_values = theta_values
		self.sample_shape = theta_values[0].shape

	def create_cluster(self):
		
		cluster = KMeans(n_clusters=self.number_of_clusters, random_state=0)

		self.theta_cluster = cluster

	def map_indexes(self):
		self.labels = self.theta_cluster.labels_

		idx_map = {}

		for i in range(len(self.labels)):
			label = self.labels[i]

			if(label not in idx_map):
				idx_map[label] = []

			idx_map[label].append(i)

		self.index_map = idx_map

	def fit(self):
		self.theta_cluster = self.theta_cluster.fit(self.theta_values)
		self.map_indexes()

	def predict(self,sample,multiple = False):

		if(not multiple):
			if(sample.shape != self.sample_shape):
				print("Shape Error.")
				return

			sample = np.array(sample).reshape(1,-1)

		else:
			for i in sample:
				if(i.shape != self.sample_shape):
					print("Shape Error.")
					return

			sample = sample.reshape(-1,self.sample_shape)

		predictions = self.theta_cluster.predict(sample)

		return predictions

	def return_cluster_centroids(self):
		return self.theta_cluster.cluster_centers_

	def return_cluster_elements(self,label):
		return self.index_map[label]

def intersection(a,b):
	c = [value for value in a if value in b] 
	return c


def loadModel(model_name, sess):
	new_saver = tf.train.import_meta_graph(model_name)
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))

	return tf.get_default_graph()

def buildDataset(mnist):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = np.array([(i/255).reshape(28,28,1) for i in x_train])
    x_test = np.array([(i/255).reshape(28,28,1) for i in x_test])
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    y_train = np.array([tf.keras.utils.to_categorical(i,num_classes = 10).reshape(10,1) for i in y_train])
    y_test = np.array([tf.keras.utils.to_categorical(i,num_classes = 10).reshape(10,1) for i in y_test])
    
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':

	sess = tf.Session()
	graph = loadModel('mnist_model-9.meta', sess)

	# load dataset
	x_train, y_train, x_test, y_test = buildDataset(mnist)	
	
	train_ratio = 0.7
	num_train = int(len(x_train) * train_ratio)
	x_valid = x_train[num_train:]
	y_valid = y_train[num_train:]
	x_train = x_train[:num_train]
	y_train = y_train[:num_train]
	
	encoding_cluster = EncodingCluster(8,"encoded_forms_senn_mnist.pickle")
	encoding_cluster.read_data()
	encoding_cluster.create_cluster()
	encoding_cluster.fit()

	theta_cluster = ThetaCluster(8,10,"theta_scores_senn_mnist.pickle")
	theta_cluster.read_data()
	theta_cluster.create_cluster()
	theta_cluster.fit()

	x = graph.get_tensor_by_name('input:0')
	y = graph.get_tensor_by_name('output:0')

	theta = graph.get_tensor_by_name('theta:0')
	encoder = graph.get_tensor_by_name('encoder:0')
	prediction = graph.get_tensor_by_name('prediction:0')

	while(True):
		ip_sample = int(input("Enter an index for test image:"))
		
		test_sample_encoding = np.zeros(encoding_cluster.sample_shape)
		test_sample_theta = np.zeros(theta_cluster.sample_shape)
		test_sample_encoding, test_sample_theta = sess.run([encoder, theta], feed_dict={x:x_test[ip_sample:ip_sample+1]})

		test_sample_theta = test_sample_theta.reshape(theta_cluster.number_of_clusters*theta_cluster.number_of_classes,)
		output_encoding_cluster = encoding_cluster.predict(test_sample_encoding[0][0])[0]
		output_theta_cluster = theta_cluster.predict(test_sample_theta)[0]

		encoding_input_images = encoding_cluster.return_cluster_elements(output_encoding_cluster)
		theta_input_images = theta_cluster.return_cluster_elements(output_theta_cluster)

		grounding_elements = intersection(encoding_input_images,theta_input_images)
		import matplotlib.pyplot as plt

		plt.imshow(x_test[ip_sample].reshape(28,28))
		plt.show()
		
		for i in range(6):
			plt.imshow(x_train[grounding_elements[i]].reshape(28,28))
			plt.show()
	