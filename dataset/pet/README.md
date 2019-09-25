# The Oxford-IIIT Pet Dataset Uploader

This code is an example of how to use Datalake and Datasets on ABEJA Platform. 

link: http://www.robots.ox.ac.uk/~vgg/data/pets/

# Usage

To create object detection dataset, run this command.

```
python upload_dataset.py -o 123456789 --datalake_name "pet-dataset" --dataset_name "pet-dataset" --format_file dataset_detection.json --attribute_type detection --max_workers 4
```

To create object classification dataset, run this command.

```
python upload_dataset.py -o 123456789 --datalake_name "pet-dataset" --dataset_name "pet-dataset" --format_file dataset_classification.json --attribute_type classification --max_workers 4
```