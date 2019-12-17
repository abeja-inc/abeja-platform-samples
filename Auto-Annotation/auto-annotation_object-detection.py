import io
import requests
import json
from urllib.parse import urljoin
from tqdm import tqdm
from abeja.datalake import Client as DatalakeClient

# set deployment url
deployment_url = 'https://abeja-annotation-demo.api.abeja.io/deployments/1874628703045/test'

# set credential
organization_id = '1698459253990'
user_id = 'user-1782414544736'
personal_access_token = '3dc265c5918b51b83582f6069fbd665243a456d6'

credential = {
    'user_id': user_id,
    'personal_access_token': personal_access_token
}

# set credential for preinference
ANNOTATION_API = 'https://annotation-tool.abeja.io'
ANNOTATION_PROJECT_ID = '5378'
ANNOTATION_ORGANIZATION_ID = '96'

headers = {
    'api-access-user-id': "2587",
    'api-access-token': "VDXU7C8A"
}

cls2id = {'aeroplane': 0,
          'bicycle': 1,
          'bird': 2,
          'boat': 3,
          'bottle': 4,
          'bus': 5,
          'car': 6,
          'cat': 7,
          'chair': 8,
          'cow': 9,
          'diningtable': 10,
          'dog': 11,
          'horse': 12,
          'motorbike': 13,
          'person': 14,
          'pottedplant': 15,
          'sheep': 16,
          'sofa':  17,
          'train': 18,
          'tvmonitor': 19
          }

# get DataLake channel id
organization_url = urljoin(ANNOTATION_API, "/api/v1/organizations/{}/projects/{}".format(ANNOTATION_ORGANIZATION_ID,
                                                                                         ANNOTATION_PROJECT_ID))
res = requests.get(organization_url, headers=headers)
res.raise_for_status()
channel_id = res.json()['data_lake_channels'][0]['channel_id']
client = DatalakeClient(organization_id=organization_id, credential=credential)
datalake_channel = client.get_channel(channel_id)

# get annotation task url
task_url = urljoin(ANNOTATION_API, "/api/v1/organizations/{}/projects/{}/tasks/".format(ANNOTATION_ORGANIZATION_ID,
                                                                                        ANNOTATION_PROJECT_ID))
page = 1

while True:
    res = requests.get(task_url, headers=headers, params={'page': page})
    res.raise_for_status()
    res_body = res.json()

    # check if project has data
    if len(res_body) == 0:
        break

    for task in tqdm(res_body):
        metadata = task['metadata'][0]
        # load file from DataLake channel
        file = datalake_channel.get_file(metadata['file_id'])
        content_type = file.get_file_info()['content_type']
        img_io = io.BytesIO(file.get_content())

        try:

            while True:
                try:
                    res = requests.post(deployment_url, data=img_io, headers={'Content-Type': content_type},
                                    auth=(user_id, personal_access_token))
                    result = res.json()
                    print(result)

                    break

                except json.decoder.JSONDecodeError:
                    print('JSON error. Retry request.')


            boxes = result['boxes']
            classes = result['classes']
            scores = result['scores']

            information = []

            for box, cls, score in zip(boxes, classes, scores):
                if float(score) >= 0.6:
                    information.append({
                        'rect': box,
                        'classes': [
                            {
                                'id': cls2id[cls],
                                'name': cls,
                                'category_id': 0
                            }
                        ]
                    })

            if len(information) == 0:
                continue

            print(information)

            # post predicted result to annotation tool
            preinference_url = urljoin(task_url, "{}/preinferences".format(str(task['id'])))
            res = requests.post(preinference_url, json={'information': information}, headers=headers)

        except KeyError:
            print(result['status'])
            print('Fail to read {}'.format(file.uri))

    page = page + 1
