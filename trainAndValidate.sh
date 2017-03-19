#!/bin/bash
source PathDefine.bash
# bazel-bin/im2txt/train   --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256"   --inception_checkpoint_file="${INCEPTION_CHECKPOINT}"   --train_dir="${MODEL_DIR}/train"   --train_inception=false   --number_of_steps=10
# bazel-bin/im2txt/run_inference   --checkpoint_path=${CHECKPOINT_DIR}   --vocab_file=${VOCAB_FILE}   --input_files=${IMAGE_FILE}   --image_dir="${IMAGE_DIR}"   --validateGlobal=${VALIDATEGLOBAL}
# python ~/coco-caption/cocoEvaluate.py >> val2014scores.txt
# bazel-bin/im2txt/train   --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256"   --inception_checkpoint_file="${INCEPTION_CHECKPOINT}"   --train_dir="${MODEL_DIR}/train"   --train_inception=false   --number_of_steps=20
# bazel-bin/im2txt/run_inference   --checkpoint_path=${CHECKPOINT_DIR}   --vocab_file=${VOCAB_FILE}   --input_files=${IMAGE_FILE}   --image_dir="${IMAGE_DIR}"   --validateGlobal=${VALIDATEGLOBAL}
# python ~/coco-caption/cocoEvaluate.py >> val2014scores.txt
TRAIN_STEP=5000
while [  $TRAIN_STEP -lt 500000 ]; do
 echo -e 'Train step at:' $TRAIN_STEP '\n' >> val2014scores.txt
 bazel-bin/im2txt/train   --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256"   --inception_checkpoint_file="${INCEPTION_CHECKPOINT}"   --train_dir="${MODEL_DIR}/train"   --train_inception=false   --number_of_steps=${TRAIN_STEP}
 bazel-bin/im2txt/run_inference   --checkpoint_path=${CHECKPOINT_DIR}   --vocab_file=${VOCAB_FILE}   --input_files=${IMAGE_FILE}   --image_dir="${IMAGE_DIR}"   --validateGlobal=${VALIDATEGLOBAL}
 python ~/coco-caption/cocoEvaluate.py >> val2014scores.txt
 echo -e '=================================================\n\n' >> val2014scores.txt
 let TRAIN_STEP=TRAIN_STEP+5000
done
