import base64
import io
import requests
from urllib.parse import urljoin
from tqdm import tqdm
import time
import yaml
from PIL import Image

from abeja.datalake import Client as DatalakeClient

# set deployment_url
deployment_url = 'XXXXXXXX'

# get credential information
yaml_dict = yaml.load(open('secret.yaml').read(), Loader=yaml.SafeLoader)

# set credential
organization_id = yaml_dict['organization_id']
user_id = yaml_dict['user_id']
personal_access_token = yaml_dict['personal_access_token']

credential = {
    'user_id': user_id,
    'personal_access_token': personal_access_token
}

# set credential for preinference
annotation_user_id = yaml_dict['annotation_user_id']
annotation_access_token = yaml_dict['annotation_access_token']
annotation_organization_id = yaml_dict['annotation_organization_id']
annotation_project_id = yaml_dict['annotation_project_id']

headers = {
    'api-access-user-id': annotation_user_id,
    'api-access-token': annotation_access_token
}

# get DataLake channel id in and out
annotation_api = 'https://annotation-tool.abeja.io'
project_url = urljoin(annotation_api, '/api/v1/organizations/{}/projects/{}'.format(annotation_organization_id,
                                                                                    annotation_project_id))
res_project_url = requests.get(project_url, headers=headers)
res_project_url.raise_for_status()
channel_id_in = res_project_url.json()['data_lake_channels'][0]['channel_id']
channel_id_out = res_project_url.json()['result_data_lake_channel']['channel_id']


def main():
    """auto-annotation for image segmentation"""
    page = 1
    delay = 10

    while True:
        # get annotation tasks
        task_url = urljoin(annotation_api, '/api/v1/organizations/{}/projects/{}/tasks/'.format(
            annotation_organization_id, annotation_project_id))
        res_task_url = requests.get(task_url,
                                    headers=headers,
                                    params={'page': page})
        res_task_url.raise_for_status()

        # check if project has data
        if len(res_task_url.json()) == 0:
            break

        for task in tqdm(res_task_url.json()):
            # load image from DataLake channel-in
            client = DatalakeClient(organization_id=organization_id, credential=credential)
            channel = client.get_channel(channel_id_in)
            metadata = task['metadata'][0]
            input_img = channel.get_file(metadata['file_id'])
            content_type = input_img.get_file_info()['content_type']
            input_img_io = io.BytesIO(input_img.get_content())

            # post image to the model
            res_deployment_url = requests.post(deployment_url,
                                               data=input_img_io,
                                               headers={'Content-Type': content_type},
                                               auth=(user_id, personal_access_token))
            res_deployment_url.raise_for_status()

            # get predicted results
            labels = res_deployment_url.json()['labels'][0]
            b64 = res_deployment_url.json()['result']
            output_img_io = io.BytesIO(base64.b64decode(b64))

            # convert black color to transparency
            img = Image.open(output_img_io).convert('RGBA')
            pixdata = img.load()

            width, height = img.size
            for y in range(height):
                for x in range(width):
                    if pixdata[x, y] == (0, 0, 0, 255):
                        pixdata[x, y] = (0, 0, 0, 0)

            modified_img_io = io.BytesIO()
            img.save(modified_img_io, format='PNG')

            # upload output image to Datalake channel-out
            upload_url = 'https://api.abeja.io//channels/{}/upload'.format(channel_id_out)
            res_upload_url = requests.post(upload_url,
                                           data=modified_img_io.getvalue(),
                                           headers={'Content-type': 'image/png'},
                                           auth=(user_id, personal_access_token))
            res_upload_url.raise_for_status()
            file_id = res_upload_url.json()['file_id']

            # get file url
            file_info_url = 'https://api.abeja.io//channels/{}/{}'.format(channel_id_out, file_id)
            res_file_info_url = requests.get(file_info_url,
                                             auth=(user_id, personal_access_token))
            res_file_info_url.raise_for_status()
            file_url = res_file_info_url.json()['download_url']

            # FIXME multi label
            information = [{
                'class': labels['label'],
                'color': labels['color'],
                'file_id': file_id,
                'file_url': file_url,
                'id': labels['label_id']}
            ]

            # register predicted result to annotation tool
            preinference_url = urljoin(task_url, "{}/preinferences".format(str(task['id'])))
            res_preinference_url = requests.post(preinference_url,
                                                 json={'information': information},
                                                 headers=headers)
            res_preinference_url.raise_for_status()
            print(res_preinference_url.json())
            time.sleep(delay)

        page = page + 1


if __name__ == '__main__':
    main()
