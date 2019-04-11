#!/bin/bash

mkdir tmp

echo "download datasets"
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P ./tmp
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P ./tmp
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P ./tmp

echo "extract datasets"
tar xvf ./tmp/VOCtrainval_06-Nov-2007.tar -C ./tmp
tar xvf ./tmp/VOCtrainval_11-May-2012.tar -C ./tmp
tar xvf ./tmp/VOCtest_06-Nov-2007.tar -C ./tmp

echo "create channels"
VOC2007_CHANNEL_ID=`abeja datalake create-channel --name "PascalVOC-2007" --description "PascalVOC2007 data" | jq -r '.channel.channel_id'`
VOC2012_CHANNEL_ID=`abeja datalake create-channel --name "PascalVOC-2012" --description "PascalVOC2012 data" | jq -r '.channel.channel_id'`

echo "upload datasets"
abeja datalake upload --channel_id ${VOC2007_CHANNEL_ID} --recursive ./tmp/VOCdevkit/VOC2007/JPEGImages/
abeja datalake upload --channel_id ${VOC2012_CHANNEL_ID} --recursive ./tmp/VOCdevkit/VOC2012/JPEGImages/
