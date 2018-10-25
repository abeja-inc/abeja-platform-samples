#!/bin/bash

wget https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip -O SegNet-Tutorial-master.zip
unzip ./SegNet-Tutorial-master.zip

TRAIN_CHANNEL_ID=`abeja datalake create-channel --name CamVid-train --description CamVid-train | jq -r '.channel.channel_id'`
VAL_CHANNEL_ID=`abeja datalake create-channel --name CamVid-val --description CamVid-val | jq -r '.channel.channel_id'`
TEST_CHANNEL_ID=`abeja datalake create-channel --name CamVid-test --description CamVid-test | jq -r '.channel.channel_id'`

TRAIN_LABEL_CHANNEL_ID=`abeja datalake create-channel --name CamVid-train-label --description CamVid-train-label | jq -r '.channel.channel_id'`
VAL_LABEL_CHANNEL_ID=`abeja datalake create-channel --name CamVid-val-label --description CamVid-val-label | jq -r '.channel.channel_id'`
TEST_LABEL_CHANNEL_ID=`abeja datalake create-channel --name CamVid-test-label --description CamVid-test-label | jq -r '.channel.channel_id'`

abeja datalake upload --channel_id ${TRAIN_CHANNEL_ID} --recursive ./SegNet-Tutorial-master/CamVid/train/
abeja datalake upload --channel_id ${VAL_CHANNEL_ID} --recursive ./SegNet-Tutorial-master/CamVid/val/
abeja datalake upload --channel_id ${TEST_CHANNEL_ID} --recursive ./SegNet-Tutorial-master/CamVid/test/

abeja datalake upload --channel_id ${TRAIN_LABEL_CHANNEL_ID} --recursive ./SegNet-Tutorial-master/CamVid/trainannot/
abeja datalake upload --channel_id ${VAL_LABEL_CHANNEL_ID} --recursive ./SegNet-Tutorial-master/CamVid/valannot/
abeja datalake upload --channel_id ${TEST_LABEL_CHANNEL_ID} --recursive ./SegNet-Tutorial-master/CamVid/testannot/
