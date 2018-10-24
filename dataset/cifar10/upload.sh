#!/bin/bash

wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzvf cifar-10-python.tar.gz
python3 extract.py

CHANNEL_ID=`abeja datalake create-channel --name CIFAR10 --description CIFAR10 | jq -r '.channel.channel_id'`
TEST_CHANNEL_ID=`abeja datalake create-channel --name CIFAR10-test --description CIFAR10 | jq -r '.channel.channel_id'`

labels=(airplane automobile bird cat deer dog frog horse ship truck)

for label in ${labels[@]}
do
  abeja datalake upload --channel_id ${CHANNEL_ID} --metadata label:${label} --recursive ./train/${label}/
  abeja datalake upload --channel_id ${TEST_CHANNEL_ID} --metadata label:${label} --recursive ./test/${label}/
done
