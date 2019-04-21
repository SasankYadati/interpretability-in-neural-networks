import numpy as np
import pickle
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.misc import toimage
from fpdf import FPDF as f


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
	new_saver.restore(sess, tf.train.latest_checkpoint('static/'))

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

def noisy(image,rec_depth,prediction,x):
  if(rec_depth==50):
    return image,False

  vals = len(np.unique(image))
  vals = 2 ** np.ceil(np.log2(vals))
  noisy_ = np.random.poisson(image * vals) / float(vals)
  if(np.argmax(sess.run(prediction,feed_dict = {x : image.reshape(1,28,28,1)})) != np.argmax(sess.run(prediction,feed_dict = {x : noisy_.reshape(1,28,28,1)}))):
     return noisy(image,rec_depth+1,prediction,x)
  return noisy_,True

def output_class_prediction(encodings,number_of_classes):
	plt.figure()
	print(encodings)
	filename = "static/"+"output_class_pred"+str(idx)+".png"
	y_ticks = ["Output Class# "+str(i) for i in range(number_of_classes)]
	fig, ax = plt.subplots()
	# Example data
	concepts = tuple(y_ticks)
	y_pos = np.arange(10)
	performance = encodings
	ax.barh(y_pos, performance, align='center',color='green', ecolor='black')
	ax.set_yticks(y_pos)
	ax.set_yticklabels(concepts)
	ax.invert_yaxis()  # labels read top-to-bottom
	ax.set_xlabel('Probabilities for each output class')
	ax.set_title('Output Class Scores')
	plt.savefig(filename)

	return filename

def save_image(img,noisy=False):
	if(noisy):
		filename = "static/"+"image_noisy"+str(idx)+".png"
	else:
		filename = "static/"+"image"+str(idx)+".png"
	im = toimage(img.reshape(28,28))
	im.save(filename)
	return filename

def mse(exp1, exp2):

  err = np.sum((exp1 - exp2) ** 2)
  err /= 16

  err = err ** 0.5
  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err

def compare(image,noisy_image,number_of_concepts,number_of_classes,predictions,theta_values,theta_values_noisy,encoded_form,encoded_form_noisy):
	theta_values = theta_values.reshape(number_of_concepts,number_of_classes)
	theta_values_noisy = theta_values_noisy.reshape(number_of_concepts,number_of_classes)
	pred = np.argmax(predictions)
	#print(pred)
	explanation_values = theta_values[:,pred]
	explanation_values_noisy = theta_values_noisy[:,pred]

	#print(explanation_values,explanation_values_noisy)
	concepts_relevance = {}

	for i in range(number_of_concepts):
	  concepts_relevance[i] = [explanation_values[i],explanation_values_noisy[i]]

	error_theta = mse(explanation_values,explanation_values_noisy)
	error_encoded_form = mse(encoded_form,encoded_form_noisy)

	return concepts_relevance,error_theta,error_encoded_form

def draw_plot(c_r):
  plt.figure()
  print(c_r)
  filename = "static/"+"concepts_relevance_scores"+str(idx)+".png"
  null = []
  for i in c_r:
    if(c_r[i] == [0.0,0.0]):
      null.append(i)
  for i in null:
    del c_r[i]
  y1 = [c_r[i][0] for i in c_r]
  y2 = [c_r[i][1] for i in c_r]
  x1 = [float(i) for i in range(len(y1))]
  x2 = [float(i)+0.2 for i in range(len(y1))]
  ax = plt.subplot(111)
  print(y1,y2)
  ax.bar(x1, y1,width=0.2,color='b',align='center')
  ax.bar(x2, y2,width=0.2,color='g',align='center')
  ax.autoscale(tight=True)

  ax.set_xlabel('Concept Number')
  ax.set_ylabel('Concept Relevance Score')
  ax.set_title('Concept Relevance Scores for Original And Noisy Image')

  plt.savefig(filename)

  return filename

def draw_plot_encodings(c1,c2):
  plt.figure()
  x1 = [float(i) for i in range(len(c1))]
  x2 = [float(i)+0.2 for i in range(len(c1))]
  ax = plt.subplot(111)

  ax.bar(x1, c1,width=0.2,color='b',align='center')
  ax.bar(x2, c2,width=0.2,color='g',align='center')

  ax.set_xlabel('Concept Number')
  ax.set_ylabel('Concept Encoding Score')
  ax.set_title('Encoding Scores for Original And Noisy Image')

  filename="static/"+"encodings_noisy_and_original"+str(idx)+".png"
  plt.savefig(filename)
  #plt.show()

  return filename

def return_heat_map(mat,heading,number_of_classes,number_of_concepts):
	plt.figure()
	fig,ax = plt.subplots(figsize=(12,7))
	plt.title(heading)

	y_ticks = ["Output Class# "+str(i) for i in range(number_of_classes)]
	ax.set_yticks(range(10))
	ax.set_yticklabels(y_ticks, rotation='horizontal', fontsize=18)

	x_ticks = ["Concept#"+str(i) for i in range(number_of_concepts)]
	ax.set_xticks(range(8))
	ax.set_xticklabels(x_ticks, rotation='horizontal', fontsize=18)


	#ax.axis('off')
	#for i in range(number_of_concepts):
	sns.heatmap(mat,fmt="",cmap="RdYlGn",linewidths = 0.30,ax =ax)
	filename = "static/"+heading+str(idx)+".png"

	plt.savefig(filename)
	return filename
def get_matrix(encoded_form,theta_values,number_of_classes,number_of_concepts):
	theta_values = theta_values.reshape(number_of_concepts,number_of_classes)
	mat = []
	for i in range(number_of_classes):
	  l = ((encoded_form[0]*theta_values[:,i])[0])
	  mat.append(l)
	mat = np.array(mat).reshape(10,8)

	return mat

def get_similar_images_image(l,concept_images,x_train):
	plt.figure()
	f, axarr = plt.subplots(l,l,figsize=(20,10))
	k=0
	m=0
	for i in range(l):
	  for j in range(l):
	    axarr[k,j].imshow(x_train[concept_images[m]].reshape(28,28))
	    m+=1
	  k+=1
	filename = "static/"+"grounding_elements"+str(idx)+".png"
	plt.savefig(filename)

	return filename
def create(image_path,noisy_image_path,output_class_scores1_path,concepts,concepts_relevance_scores,encodings_scores,theta_original,theta_noisy,theta_diff,grounding_elements_filename):
	pdf = f()
	pdf.add_page()

	pdf.set_font("Arial", size=24)
	pdf.cell(200,10,txt="Explanations for the Image")

	pdf.set_font("Arial", size=14)
	pdf.ln(20)
	pdf.cell(200,10,txt="Original Image")
	pdf.ln(20)
	pdf.image(image_path,w=100)

	pdf.ln(20)
	pdf.cell(200,10,txt="Output Class Scores")
	pdf.ln(20)
	pdf.image(output_class_scores1_path,w=100)

	s = "Given below are the images that maximally activate each concept."
	s1 = "Each row corresponds to a particular concept."
	pdf.ln(20)
	pdf.cell(200,10,txt=s)
	pdf.ln(5)
	pdf.cell(200,10,txt=s1)
	pdf.ln(20)
	pdf.image(concepts,w=150)

	pdf.ln(20)
	pdf.cell(200,10,txt="Noisy Image")
	pdf.ln(20)
	pdf.image(noisy_image_path,w=100)

	pdf.ln(40)
	pdf.cell(200,10,txt ="Encodings Scores for original image and noisy image for each concept.")
	pdf.ln(20)
	pdf.image(encodings_scores,w=150,h=100)

	pdf.ln(20)
	pdf.cell(200,10,txt ="Concept Relevance Scores for original image and noisy image.")
	pdf.ln(20)
	pdf.image(concepts_relevance_scores,w=150,h=80)

	pdf.ln(20)
	pdf.cell(200,10,txt="Theta Matrix for the Original Image as a heat map")
	pdf.ln(20)
	pdf.image(theta_original,w=200,h=100)

	pdf.ln(20)
	pdf.cell(200,10,txt="Theta Matrix for the Noisy Image as a heat map")
	pdf.ln(20)
	pdf.image(theta_noisy,w=200,h=100)

	pdf.ln(20)
	pdf.cell(200,10,txt="Theta Matrix Difference between Original and Noisy Image as a heat map")
	pdf.ln(20)
	pdf.image(theta_diff,w=200,h=100)

	pdf.ln(160)
	pdf.cell(200,10,txt="Grounding Images which were similar to the input test image.")
	pdf.ln(10)
	pdf.cell(200,10,txt="Model learnt to predict from this class from these.")
	pdf.ln(20)
	pdf.image(grounding_elements_filename,w=200,h=100)

	pdf.output("static/test.pdf")
idx = None
sess = None
def getPDF(x_,index):
	global idx
	idx = index
	global sess
	print(sess)
	sess = tf.Session()
	graph = loadModel('static/mnist_model-9.meta', sess)
	print(sess)
	# load dataset
	x_train, y_train, x_test, y_test = buildDataset(mnist)

	train_ratio = 0.7
	num_train = int(len(x_train) * train_ratio)
	x_valid = x_train[num_train:]
	y_valid = y_train[num_train:]
	x_train = x_train[:num_train]
	y_train = y_train[:num_train]

	encoding_cluster = EncodingCluster(8,"static/encoded_forms_senn_mnist.pickle")
	encoding_cluster.read_data()
	encoding_cluster.create_cluster()
	encoding_cluster.fit()

	theta_cluster = ThetaCluster(8,10,"static/theta_scores_senn_mnist.pickle")
	theta_cluster.read_data()
	theta_cluster.create_cluster()
	theta_cluster.fit()

	x = graph.get_tensor_by_name('input:0')
	y = graph.get_tensor_by_name('output:0')

	theta = graph.get_tensor_by_name('theta:0')
	encoder = graph.get_tensor_by_name('encoder:0')
	prediction = graph.get_tensor_by_name('prediction:0')

	test_sample_encoding, test_sample_theta = sess.run([encoder, theta], feed_dict={x:x_})

	print(test_sample_encoding)

	test_sample_theta = test_sample_theta
	output_encoding_cluster = encoding_cluster.predict(test_sample_encoding[0][0])[0]
	output_theta_cluster = theta_cluster.predict(test_sample_theta.reshape(theta_cluster.number_of_classes*theta_cluster.number_of_clusters))[0]

	noisy_image, _ = noisy(x_[0],20,prediction,x)
	noisy_sample_encoding, noisy_sample_theta = sess.run([encoder, theta], feed_dict={x:[noisy_image]})

	print(x_[0].shape)
	original_image_filename = save_image(x_[0])
	noisy_image_filename = save_image(noisy_image,noisy=True)

	predictions = sess.run(prediction,feed_dict={x:x_})
	predictions = [i for j in range(10) for i in predictions[0][j]]
	print(predictions)
	output_class = np.argmax(predictions)
	output_scores_filename = output_class_prediction(predictions,theta_cluster.number_of_classes)
	concepts_image_filename = "static/concepts1.png"

	c_r,mse_theta,mse_encoded = compare(x_[0],noisy_image,theta_cluster.number_of_clusters,theta_cluster.number_of_classes,predictions,test_sample_theta,noisy_sample_theta,test_sample_encoding[0][0],noisy_sample_encoding[0][0])
	concepts_relevance_scores_filename = draw_plot(c_r)
	encodings_noisy_and_original_filename = draw_plot_encodings(test_sample_encoding[0][0],noisy_sample_encoding[0][0])

	mat_original = get_matrix(test_sample_encoding,test_sample_theta,theta_cluster.number_of_classes,theta_cluster.number_of_clusters)
	mat_noisy = get_matrix(noisy_sample_encoding,noisy_sample_theta,theta_cluster.number_of_classes,theta_cluster.number_of_clusters)

	theta_original_filename = return_heat_map(mat_original,"Theta Matrix for Original Image",theta_cluster.number_of_classes,theta_cluster.number_of_clusters)
	theta_noisy_filename = return_heat_map(mat_noisy,"Theta Matrix for Noisy Image",theta_cluster.number_of_classes,theta_cluster.number_of_clusters)
	mat_diff = mat_original - mat_noisy
	theta_diff_filename = return_heat_map(mat_diff,"Difference between Original and Noisy Theta Values",theta_cluster.number_of_classes,theta_cluster.number_of_clusters)

	encoding_input_images = encoding_cluster.return_cluster_elements(output_encoding_cluster)
	theta_input_images = theta_cluster.return_cluster_elements(output_theta_cluster)

	grounding_elements = intersection(encoding_input_images,theta_input_images)
	l=16
	grounding_elements_sample = grounding_elements[0:l]
	grounding_elements_filename = get_similar_images_image(4,grounding_elements_sample,x_train)

	create(original_image_filename,noisy_image_filename,output_scores_filename,concepts_image_filename,concepts_relevance_scores_filename,encodings_noisy_and_original_filename,theta_original_filename,theta_noisy_filename,theta_diff_filename,grounding_elements_filename)
	return [test_sample_encoding[0][0],original_image_filename,noisy_image_filename,output_scores_filename,concepts_image_filename,encodings_noisy_and_original_filename,theta_diff_filename,grounding_elements_filename,output_class,mse_theta,mse_encoded]
