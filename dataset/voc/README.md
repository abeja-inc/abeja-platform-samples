# Upload instruction of Pascal VOC Dataset to ABEJA Platform 

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

以下のコマンドでPascalVOCデータセットをダウンロードし、画像をPlatform上にアップロードします。

```
$ cd voc
$ upload.sh
```

## アノテーションデータの登録

### Datalake IDの確認

先ほどアップロードしたデータの`channel_id`を確認します。ここでは、`PascalVOC-2007_ID`と`PascalVOC-2012_ID`とします。

```
$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="PascalVOC-2007")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="PascalVOC-2012")'
```

続いて、アノテーションデータをアップロードします。

```
$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -c {PascalVOC-2007_ID} \
          -d PascalVOC2007-trainval \
          --split trainval \
          --year 2007

$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -c {PascalVOC-2012_ID} \
          -d PascalVOC2012-trainval \
          --split trainval \
          --year 2012
          
$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -c {PascalVOC-2007_ID} \
          -d PascalVOC2007-test \
          --split test \
          --year 2007
```
