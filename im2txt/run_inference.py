

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os, os.path
import json

import imghdr

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

img_dir = '/home/superNLP/usb_hdd/cocodata/raw-data/pic100/'
#dict_dir = '/home/superNLP/usb_hdd/cocodata/raw-data/annotations/captions_val2014_filename_id.json'
output_dir = 'captions_pic100_showandtell_results.json'


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  '''
  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)'''

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    print("going to print!")  
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names,values):
      print(k,v)

    
    file_writer = tf.summary.FileWriter('/home/superNLP/ours/im2txt_attention/tesnboard', sess.graph)
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    imgId_cap = []
    dict_file = open(dict_dir, 'r')
    dict_data = json.load(dict_file)
    num = 0
    output_file = open(output_dir,"w")
    for filename in os.listdir(img_dir):
      #filename="COCO_val2014_000000320612.jpg"
      print(img_dir+filename, 'filepath')
      print(num)
      num += 1
      if num>10:
          break
      if(filename != '.' and filename != '..'):
          #filename="/home/superNLP/usb_hdd/cocodata/"
          with tf.gfile.GFile(img_dir+filename, "r") as f:
            image = f.read()
          '''
          if False:#imghdr.what(img_dir+filename)!='jpeg':
              pngCount=pngCount+1
              print("wrong format :",imghdr.what(img_dir+filename))
              print("pngCount :",pngCount)
          else:
          '''
          captions = generator.beam_search(sess, image)
          #print("Captions for image %s:" % os.path.basename(filename))
          for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            print('caption predicted:', sentence)
            #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            temp = {}
            temp["image_id"] = dict_data[filename]
            temp["caption"] = sentence[0:len(sentence)-2]
            imgId_cap.append(temp)
            break
    #output_file.write(json.dumps(imgId_cap))
    file_writer.add_graph(sess.graph)
    file_writer.close()
if __name__ == "__main__":
  tf.app.run()
