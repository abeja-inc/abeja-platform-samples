# Upload instruction of CIFAR10 dataset to ABEJA Platform 

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

以下のコマンドでCIFAR10データセットをダウンロードし、画像をPlatform上にアップロードします。

```
$ cd cifar10
$ upload.sh
```

## アノテーションデータの登録

### Datalake IDの確認

先ほどアップロードしたデータの`channel_id`を確認します。ここでは、`CIFAR10_ID`と`CIFAR10-test_ID`とします。

```
$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CIFAR10")'

$ curl -X GET https://api.abeja.io/organizations/{ORGANIZATION_ID}/channels \
       -u user-{USER_ID}:{TOKEN} | jq '.channels[] | select (.name=="CIFAR10-test")'
```

続いて、アノテーションデータをアップロードします。

```
$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -c {CIFAR10_ID} \
          -d CIFAR10

$ python import_dataset_from_datalake.py \
          -o {ORGANIZATION_ID} \
          -c {CIFAR10-test_ID} \
          -d CIFAR10-test
```
