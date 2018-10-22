#!/bin/bash

mkdir tmp

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P ./tmp
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P ./tmp
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P ./tmp

tar xvf ./tmp/VOCtrainval_06-Nov-2007.tar -C ./tmp
tar xvf ./tmp/VOCtrainval_11-May-2012.tar -C ./tmp
tar xvf ./tmp/VOCtest_06-Nov-2007.tar -C ./tmp

VOC2007_CHANNEL_ID=`abeja datalake create-channel --name "PascalVOC-2007" --description "PascalVOC2007 data" | jq -r '.channel.channel_id'`
VOC2012_CHANNEL_ID=`abeja datalake create-channel --name "PascalVOC-2012" --description "PascalVOC2012 data" | jq -r '.channel.channel_id'`

abeja datalake upload --channel_id ${VOC2007_CHANNEL_ID} --recursive ./tmp/VOCdevkit/VOC2007/JPEGImages/
abeja datalake upload --channel_id ${VOC2012_CHANNEL_ID} --recursive ./tmp/VOCdevkit/VOC2012/JPEGImages/

VOC2007_TRAINVAL_DATASET_ID=`abeja dataset create-dataset --name "VOCtrainval-2007" --type detection --props dataset.json | jq -r '.dataset_id'`
VOC2007_TEST_DATASET_ID=`abeja dataset create-dataset --name "VOCtest-2007" --type detection --props dataset.json | jq -r '.dataset_id'`
VOC2012_TRAINVAL_DATASET_ID=`abeja dataset create-dataset --name "VOCtrainval-2012" --type detection --props dataset.json | jq -r '.dataset_id'`

python import_dataset_from_datalake.py --channel_id ${VOC2007_CHANNEL_ID} --dataset_id ${VOC2007_TRAINVAL_DATASET_ID} --split "trainval" --year 2007
python import_dataset_from_datalake.py --channel_id ${VOC2007_CHANNEL_ID} --dataset_id ${VOC2007_TEST_DATASET_ID} --split "test" --year 2007
python import_dataset_from_datalake.py --channel_id ${VOC2012_CHANNEL_ID} --dataset_id ${VOC2012_TRAINVAL_DATASET_ID} --split "trainval" --year 2012
