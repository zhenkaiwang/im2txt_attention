# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops
from tensorflow.python.ops import variable_scope as vs
#RNNCELL
from im2txt.ops import rnn_cell_ops


tf.nn.rnn_cell=rnn_cell_ops

class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.
  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
        print("__init__")
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, sub_featrue_num, sub_feature_length].
    self.image_sub_features = None #superNLP

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    self.inputs_wa = None



  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
        print("process_image")
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
        print("build_inputs")
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  #superNLP
  # def build_sub_image_embeddings(self):
  #   """Builds the image model subgraph and generates image embeddings.

  #   Inputs:
  #     self.images

  #   Outputs:
  #     self.image_embeddings
  #   """
  #   inception_output = image_embedding.inception_v3(
  #       self.images,
  #       trainable=self.train_inception,
  #       is_training=self.is_training(),scope="InceptionV3",layer="Conv2d_4a_3x3")
  #   self.inception_variables = tf.get_collection(
  #       tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
  #   '''
  #   # Map inception output into embedding space.
  #   with tf.variable_scope("image_embedding") as scope:
  #     image_embeddings = tf.contrib.layers.fully_connected(
  #         inputs=inception_output,
  #         num_outputs=self.config.embedding_size,
  #         activation_fn=None,
  #         weights_initializer=self.initializer,
  #         biases_initializer=None,
  #         scope=scope)

  #   # Save the embedding size in the graph.
  #   tf.constant(self.config.embedding_size, name="embedding_size")
  #   '''
  #   #IncepShape = inception_output.get_shape()
  #   print(inception_output)
  #   self.image_sub_features = inception_output
  # #superNLP


  def build_image_embeddings(self):
        print("build_image_embeddings")
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output,inception_output_sub = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings
    self.image_sub_features = inception_output_sub


  def build_seq_embeddings(self):
        print("build_seq_embeddings")
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings

  def build_inputs_wa(self):
        print("build_inputs_wa")
    # size1=tf.shape(self.seq_embeddings)
 #        size2=tf.shape(self.image_sub_features)
    # padded_length = size1[1]
 #        _,padded_length_num,embedding_size=self.seq_embeddings.get_shape()
    # batch_size, sub_feature_num, sub_feature_length = self.image_sub_features.get_shape()
    # # generate RNN input: word feature + local features
    # output = tf.zeros([size1[0], size1[1],size1[2] + size2[1]*size2[2]], tf.float32)
    # index=0
    # def while_condition(i,index,img):
 #            return tf.less(i,padded_length)
    # def _body(i,index,img):
 #                sess2 = tf.Session()
        
    #   print("img")
 #                print(img)
    #   print(sess2.run(img))
 #                print("index")
 #                print(index)
    #   output[img][index] = tf.concat([tf.reshape(self.seq_embeddings[img][index], [1,-1]), tf.reshape(self.image_sub_features[img], [1,-1])],1)
    #   index=index+1
    #   return [tf.add(i,1)]

    # for img in range(batch_size):
 #                print("img for_loop")
 #                print(img)
    #   i = tf.constant(0)
    #   index=0
    #   r=tf.while_loop(while_condition,_body,[i,index,img])
    # shape_sub_list= self.image_sub_features.get_shape().as_list()
        batch_size1,padded_length_num,embedding_size=self.seq_embeddings.get_shape()
        print("batch_size1 ")
        print(batch_size1.value)
        batch_size2, sub_feature_num, sub_feature_length = self.image_sub_features.get_shape()
        print("batch_size2 ")
        print(batch_size2.value)

    shape_sub=tf.shape(self.image_sub_features)
    shape_seq=tf.shape(self.seq_embeddings)
    #image_sub_reshaped=tf.reshape(self.image_sub_features,[self.config.batch_size,1,-1])
        # image_sub_reshaped=tf.reshape(self.image_sub_features,[self.config.batch_size,1,-1])
        if self.mode=="inference":
          image_sub_reshaped=tf.reshape(self.image_sub_features,[1,1,-1])
          image_sub_tile=tf.tile(image_sub_reshaped,tf.pack([1,shape_seq[1],1]))
          print("image_sub_reshaped")
          print(image_sub_reshaped)
          print("image_sub_tile")
          print(image_sub_tile)
        else:
          image_sub_reshaped=tf.reshape(self.image_sub_features,[self.config.batch_size,1,-1])
          image_sub_tile=tf.tile(image_sub_reshaped,tf.pack([1,shape_seq[1],1]))
    
    output=tf.concat(2,[self.seq_embeddings,image_sub_tile])
    print('sub features shape')
    print(self.image_sub_features)
    print('input_wa')
    print(output)
    self.inputs_wa=output
    # return output


  def build_model(self):
        print("build_model")
	"""Builds the model.

	Inputs:
	  self.image_embeddings
	  self.seq_embeddings
	  self.target_seqs (training and eval only)
	  self.input_mask (training and eval only)

	Outputs:
	  self.total_loss (training and eval only)
	  self.target_cross_entropy_losses (training and eval only)
	  self.target_cross_entropy_loss_weights (training and eval only)
	"""
	# This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
	# modified LSTM in the "Show and Tell" paper has no biases and outputs
	# new_c * sigmoid(o).
	# lstm_cell = tf.contrib.rnn.BasicLSTMCell(
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
	# lstm_cell = rnn_cell_ops.BasicLSTMCell(
		num_units=self.config.num_lstm_units, state_is_tuple=True)
	if self.mode == "train":
	  lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
	  # lstm_cell = rnn_cell_ops.DropoutWrapper(
		  lstm_cell,
		  input_keep_prob=self.config.lstm_dropout_keep_prob,
		  output_keep_prob=self.config.lstm_dropout_keep_prob)

	with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
	  # Feed the image embeddings to set the initial LSTM state.
	  zero_state = lstm_cell.zero_state(
		  batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
	  _, initial_state = lstm_cell(self.image_embeddings, zero_state)

	  # Allow the LSTM variables to be reused.
	  lstm_scope.reuse_variables()

	  if self.mode == "inference":
		# In inference mode, use concatenated states for convenient feeding and
		# fetching.
		#tf.concat(initial_state, 1, name="initial_state")
		tf.concat(1,initial_state, name="initial_state")

		# Placeholder for feeding a batch of concatenated states.
		state_feed = tf.placeholder(dtype=tf.float32,
									shape=[None, sum(lstm_cell.state_size)],
									name="state_feed")
		#state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
		state_tuple = tf.split(split_dim=1,num_split=2,value=state_feed)

		# Run a single LSTM step.
		print("self.seq_embeddings:")
		print(self.seq_embeddings)
		# inputs_wa = self.get_inputs()
		
		lstm_outputs, state_tuple = lstm_cell(
			inputs=tf.squeeze(self.inputs_wa, squeeze_dims=[1]),
			state=state_tuple)

		# Concatentate the resulting state.
		#tf.concat(state_tuple, 1, name="state")
		tf.concat(1,state_tuple, name="state")
	  else:
		# Run the batch of sequence embeddings through the LSTM.
		# inputs_wa = self.get_inputs()

		sequence_length = tf.reduce_sum(self.input_mask, 1)
		lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
											inputs=self.inputs_wa,
											sequence_length=sequence_length,
											initial_state=initial_state,
											dtype=tf.float32,
											scope=lstm_scope)
		# lstm_outputs, _ = dynamic_rnn(cell=lstm_cell,
		#                                     inputs=self.seq_embeddings,
		#                                     sequence_length=sequence_length,
		#                                     initial_state=initial_state,
		#                                     dtype=tf.float32,
		#                                     scope=lstm_scope)        

	# Stack batches vertically.
	lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

	with tf.variable_scope("logits") as logits_scope:
	  logits = tf.contrib.layers.fully_connected(
		  inputs=lstm_outputs,
		  num_outputs=self.config.vocab_size,
		  activation_fn=None,
		  weights_initializer=self.initializer,
		  scope=logits_scope)

	if self.mode == "inference":
	  tf.nn.softmax(logits, name="softmax")
	else:
	  targets = tf.reshape(self.target_seqs, [-1])
	  weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

	  # Compute losses.
	  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
															  logits=logits)
	  batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
						  tf.reduce_sum(weights),
						  name="batch_loss")
	  #tf.losses.add_loss(batch_loss)
	  tf.contrib.losses.add_loss(batch_loss)
	  #total_loss = tf.losses.get_total_loss()
	  # with vs.variable_scope("lstm/BasicLSTMCell",reuse=True):
	  # 	fatt=tf.get_variable(name="f_att_matrix")
	  # total_loss = tf.contrib.losses.get_total_loss()+tf.nn.l2_loss(fatt)
          with vs.variable_scope("lstm/BasicLSTMCell",reuse=True):
            W1 = vs.get_variable(name="w1")
            W2 = vs.get_variable(name="w2")
            b1 = vs.get_variable(name="b1")
            b2 = vs.get_variable(name="b2") 
      reg_parm=0.1
      total_loss = tf.contrib.losses.get_total_loss()+reg_parm*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(b2))

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
        print("setup_inception_initializer")
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
        print("setup_global_step")
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        with vs.variable_scope("lstm/BasicLSTMCell",reuse=True):
            W1 = vs.get_variable(name="w1")
            W2 = vs.get_variable(name="w2")
            b1 = vs.get_variable(name="b1")
            b2 = vs.get_variable(name="b2") 
            sess = tf.Session()
            print(sess.run([W1,W2,b1,b2]))
            sess.close()

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    # self.build_sub_image_embeddings() #superNLP
    self.build_seq_embeddings()
    self.build_inputs_wa()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
