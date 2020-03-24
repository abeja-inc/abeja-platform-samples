import io
import requests
from urllib.parse import urljoin
from tqdm import tqdm

ANNOTATION_API = 'https://annotation-tool.abeja.io'

# set id for preinference
ANNOTATION_PROJECT_ID = 'XXXX'
ANNOTATION_ORGANIZATION_ID = 'XXXX'

# set credential for preinference
headers = {
    'api-access-user-id': 'XXXX',
    'api-access-token': 'XXXXX'
}

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
        try:
            # Please input label ex: {'flower': 'sunflower'}
            information = {'XXXXX': 'XXXXXX'}
            json={'information': information}
            #print(json)
            # post result to annotation tool
            preinference_url = urljoin(task_url, "{}/preinferences".format(str(task['id'])))
            res = requests.post(preinference_url, json={'information': information}, headers=headers)
            print(res)
        except KeyError:
            print('error')

    page = page + 1
