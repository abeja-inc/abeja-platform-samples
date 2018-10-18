#!/bin/bash

DATASET_ID=`abeja dataset create-dataset --name CIFAR10 --type classification --props dataset.json | jq '.dataset_id'`
abeja dataset import-from-datalake --channel_id 1578967877196 --dataset_id $DATASET_ID
