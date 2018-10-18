#!/bin/bash

abeja datalake upload --channel_id 1578967877196 --metadata label:airplane --recursive ./train/airplane/
abeja datalake upload --channel_id 1578967877196 --metadata label:automobile --recursive ./train/automobile/
abeja datalake upload --channel_id 1578967877196 --metadata label:bird --recursive ./train/bird/
abeja datalake upload --channel_id 1578967877196 --metadata label:cat --recursive ./train/cat/
abeja datalake upload --channel_id 1578967877196 --metadata label:deer --recursive ./train/deer/
abeja datalake upload --channel_id 1578967877196 --metadata label:dog --recursive ./train/dog/
abeja datalake upload --channel_id 1578967877196 --metadata label:frog --recursive ./train/frog/
abeja datalake upload --channel_id 1578967877196 --metadata label:horse --recursive ./train/horse/
abeja datalake upload --channel_id 1578967877196 --metadata label:ship --recursive ./train/ship/
abeja datalake upload --channel_id 1578967877196 --metadata label:truck --recursive ./train/truck/
