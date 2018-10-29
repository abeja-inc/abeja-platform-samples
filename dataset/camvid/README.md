# Upload instruction of CamVid dataset to ABEJA Platform 

TODO: English Documentation

## 準備

### ABEJA CLI及びSDKのインストール

pipコマンド（Pythonのパッケージ管理システム）を利用することで、下記の通りインストールすることが可能です。

```
$ curl -s https://packagecloud.io/install/repositories/abeja/platform-public/script.python.sh | bash
$ pip install abejacli abeja-sdk
```

### Organizationの設定

CLIを使用する前に、ABEJAの資格情報を設定してください。

```
abeja configure
abeja-platform-user  : 1234567890123
personal-access-token: a08jaife79ja89fjfi2l34bioat90pngiy8932ac
organization-name    : {ORGANIZATION_NAME}
[INFO]: ABEJA credentials setup completed!
```

## 画像データのアップロード

以下のコマンドでCamVidデータセットをダウンロードし、画像およびセグメンテーション画像をPlatform上にアップロードします。

```
$ cd camvid
$ upload.sh
```

## アノテーションデータの登録

### Datalake IDの確認

先ほどアップロードしたデータの`channel_id`を確認します。ここでは、`CAMVID_TRAIN_ID`、`CAMVID_VAL_ID`、
`CAMVID_TEST_ID`、`CAMVID_TRAIN_LABEL_ID`、`CAMVID_VAL_LABEL_ID`、`CAMVID_TEST_LABEL_ID`とします。

```
$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CamVid-train")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CamVid-val")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CamVid-test")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CamVid-train-label")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CamVid-val-label")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CamVid-test-label")'

```

続いて、画像とセグメンテーションを紐付けし、アノテーション付きのデータとして登録します。

```
$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -i {CAMVID_TRAIN_ID} \
          -l {CAMVID_TRAIN_LABEL_ID} \
          -d CamVid-train \
          --img_list_path ./SegNet-Tutorial-master/CamVid/train.txt

$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -i {CAMVID_VAL_ID} \
          -l {CAMVID_VAL_LABEL_ID} \
          -d CamVid-val \
          --img_list_path ./SegNet-Tutorial-master/CamVid/val.txt

$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -i {CAMVID_TEST_ID} \
          -l {CAMVID_TEST_LABEL_ID} \
          -d CamVid-test \
          --img_list_path ./SegNet-Tutorial-master/CamVid/test.txt
```
