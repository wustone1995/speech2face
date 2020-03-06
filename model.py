import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
class DataLoading:
	def __init__(self):
		self.train_csv = "./avspeech_train.csv"

	@staticmethod
	def load_ids(from_id, to_id, split='train'):
		train_csv = "./drive/My Drive/avspeech_train.csv"
		print("Loading IDs ............")
		data = pd.read_csv(train_csv, header = None, names = ["id", "start", "end", "x", "y"])
		ids =[]
		for i in range(from_id, to_id + 1):
			if (not os.path.isfile('preprocess/data/' + split + '/audio_spectrograms/' + data.loc[i, "id"] + ".pkl")):
				continue
			elif (not os.path.isfile('preprocess/data/' + split + '/speaker_video_embeddings/' + data.loc[i, "id"] + ".pkl")):
				continue
			else:
				ids.append(data.loc[i, "id"])
		print("Total " ,len(ids),  split , " IDs Loaded !! ")
		return ids

	@staticmethod
	def split_data(_ids, split = [0.8, 0.1, 0.1]):
		data = np.array(_ids)
		# np.random.shuffle(data)#train on same data everytime
		(valid, test) = (int(split[1]*len(_ids)), int(split[2]*len(_ids)))
		train =  len(_ids) - (valid + test)
		train_split, valid_split, test_split = data[0:train], data[train:train+valid], data[train+valid:]
		print("Total " ,"train:",train, "valid:",valid, "test:", test, " IDs Loaded !! ")
		return train_split, valid_split, test_split

	@staticmethod
	def load_data(_ids, split='train'):
		x_data = np.zeros((len(_ids), 598, 257, 2))
		y_data = np.zeros((len(_ids), 4096))
		for i in range(len(_ids)):
			with open('preprocess/data/' + split + '/audio_spectrograms/' +  _ids[i] + ".pkl", 'rb') as f:
				x_data[i] = pickle.load(f)
			with open('preprocess/data/' + split + '/speaker_video_embeddings/' + _ids[i] + ".pkl", 'rb') as f:
				y_data[i] = pickle.load(f)
		return x_data,y_data


class AudioEmbeddingModel:
	def __init__(self, from_id, to_id,audio_shape = (598,257,2)):
		self.from_id = from_id
		self.to_id = to_id
		def build_model(audio_shape):
			ip = tf.keras.layers.Input(shape = audio_shape)

			x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(ip)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)
			
			x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

			x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

			x = tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

			x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=2,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=2,padding="VALID")(x)

			x = tf.keras.layers.AveragePooling2D(pool_size=(6,1),strides=1,padding="VALID")(x)
			x = tf.keras.layers.ReLU()(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			flatten = tf.keras.layers.Flatten()(x)
			
			dense = tf.keras.layers.Dense(4096, activation = "relu")(flatten)
			dense = tf.keras.layers.Dense(4096)(dense)

			model = tf.keras.Model(ip, dense)
			return model
		self.model = build_model(audio_shape)
		self._lambda = 1
	
	def multi_gpu_model(self,num_gpu):
		if num_gpu > 1:
			self.model = tf.keras.utils.multi_gpu_model(self.model, gpus = num_gpu)

	def model_summary(self):
		self.model.summary()

	def l2_norm_loss_fn(self,y_true, y_pred):
		return 2 * tf.nn.l2_loss(tf.math.l2_normalize(y_true, axis=1, epsilon=1e-12) - tf.math.l2_normalize(y_pred, axis=1, epsilon=1e-12))

	def loss_fn(self,y_true, y_pred):
		return self._lambda * self.l2_norm_loss_fn(y_true,y_pred)

	def get_loss(num_samples,split='valid'):
		ids = load_ids(self.from_id, self.to_id, split)
		ids_helper = np.array(ids)
		np.random.shuffle(ids_helper)

		for i in range(int(np.ceil(len(ids_helper)/num_samples))):
			x_data, y_data = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])	  
			self.model.predict(x = x_data, y = y_train, epochs = curr_epoch + 1, batch_size = batchsize, verbose = True, initial_epoch = curr_epoch)

	def train(self,train_ids,valid_ids, batchsize, model_save_path,start_epoch = 0, num_epoch = 10, num_samples = 102 ):

		opt_fn = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, decay = 0.95/10000, amsgrad=False)
		#if less Gpu memory then try below
		# opt_fn = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

		self.model.compile(optimizer = opt_fn,loss=self.loss_fn)

		for curr_epoch in range(start_epoch, start_epoch + num_epoch):
			print("Current Epoch : ", curr_epoch)
			ids_helper = np.array(train_ids)
			np.random.shuffle(ids_helper)

			for i in range(int(np.ceil(len(ids_helper)/num_samples))):
				x_train, y_train = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])	  
				self.model.fit(x = x_train, y = y_train, epochs = curr_epoch + 1, batch_size = batchsize, verbose = True, initial_epoch = curr_epoch)

			ids_helper = np.array(valid_ids)
			Validation_loss=0
			for i in range(int(np.ceil(len(ids_helper)/num_samples))):
				X_val, Y_val = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])
				Validation_loss +=  X_val.shape[0]*self.model.evaluate(X_val, Y_val, verbose=0)
			print("Avg Validation Loss after epoch {} : {}".format(curr_epoch, Validation_loss/len(valid_ids)))
			
			if (curr_epoch % 5) == 0:
				if model_save_path.endswith('/'):
					self.model.save_weights(model_save_path + 'epoch_' + str(int(curr_epoch)) + '.h5')
				else:
					self.model.save_weights(model_save_path + '/' + 'epoch_' + str(int(curr_epoch)) + '.h5')
		

	def test(self,test_ids, batchsize, num_samples = 102 ):
		ids_helper = np.array(test_ids)
		Total_loss=0
		for i in range(int(np.ceil(len(ids_helper)/num_samples))):
			X_test, Y_test = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])
			Total_loss +=  X_test.shape[0]*self.model.evaluate(X_test, Y_test, verbose=0)
		print("Avg Test Loss : {}".format(Total_loss/len(test_ids)))
		return Total_loss/len(test_ids)
