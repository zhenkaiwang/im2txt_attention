# CNN-Attention-LSTM Image Caption Generator

A TensorFlow implementation of the image caption generator using CNN, attention and LSTM, based on Google [im2txt](https://github.com/tensorflow/models/tree/master/im2txt) *Show and Tell* model.

## Contact
Original Author: Chris Shallue (shallue@google.com)
Authors: Xiaobai Ma (maxiaoba@stanford.edu), Zhenkai Wang (zackwang@stanford.edu), Zhi Bie (zhib@stanford.edu)

## Getting Started

### Install Required Packages
First ensure that you have installed the following required packages:

* **Bazel** ([instructions](http://bazel.io/docs/install.html)).
* **TensorFlow** r0.12 or greater ([instructions](https://www.tensorflow.org/versions/master/get_started/os_setup.html)).
* **NumPy** ([instructions](http://www.scipy.org/install.html)).
* **Natural Language Toolkit (NLTK)**:
    * First install NLTK ([instructions](http://www.nltk.org/install.html)).
    * Then install the NLTK data ([instructions](http://www.nltk.org/data.html)).

### Prepare the Training Data

To train the model you will need to provide training data in native TFRecord
format. The TFRecord format consists of a set of sharded files containing
serialized `tf.SequenceExample` protocol buffers. Each `tf.SequenceExample`
proto contains an image (JPEG or PNG format), a caption and metadata such as the image
id.

Each caption is a list of words. During preprocessing, a dictionary is created
that assigns each word in the vocabulary to an integer-valued id. Each caption
is encoded as a list of integer word ids in the `tf.SequenceExample` protos.

We have provided a script to download and preprocess the [MSCOCO]
(http://mscoco.org/) image captioning data set into this format. Downloading
and preprocessing the data may take several hours depending on your network and
computer speed. Please be patient.

Before running the script, ensure that your hard disk has at least 150GB of
available space for storing the downloaded and processed data.

Modify your path variables in PathDefine.bash, then:
```shell
source PathDefine.bash

# Build the preprocessing script.
bazel build im2txt/download_and_preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco "${MSCOCO_DIR}"
```

To only build some subset of the mscoco data, first modify your path variables in PathDefine_test.bash, then:
```shell
source PathDefine_test.bash

# Build the preprocessing script.
bazel build im2txt/download_and_preprocess_mscoco_sub

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco_sub "${MSCOCO_DIR}"
```

### Download the Inception v3 Checkpoint

The model requires a pretrained *Inception v3* checkpoint file
to initialize the parameters of its image encoder submodel.

This checkpoint file is provided by the
[TensorFlow-Slim image classification library](https://github.com/tensorflow/models/tree/master/slim#tensorflow-slim-image-classification-library)
which provides a suite of pre-trained image classification models. You can read
more about the models provided by the library
[here](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models).


Run the following commands to download the *Inception v3* checkpoint.

```shell
#source PathDefine.bash or PathDefine_test.sh if not done
wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"
```


## Training a Model

### Initial Training

Run the training script.

```shell
#source PathDefine.bash or PathDefine_test.bash (for subset) if not done
# Build the model.
bazel build -c opt im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
```

## Generating Captions into Json file

Your trained model can generate captions for any JPEG or PNG image. The
following command line will generate captions for an image from the test set.

```shell
#source PathDefine.bash or PathDefine_test.bash (for subset) if not done

# Build the inference binary.
bazel build -c opt im2txt/run_inference

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run inference to generate captions into json file
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE} \
  --image_dir=${IMAGE_DIR} \
  --validateGlobal=${VALIDATEGLOBAL}
```
