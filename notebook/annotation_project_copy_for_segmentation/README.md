### 利用手順
- Annotation Toolにログイン ( https://annotation-tool-client.abeja.io/manager )
- コピーを実施したいプロジェクトのJSONファイルをダウンロード
- コピーを実施するためのプロジェクトを作成　(「プロジェクトの複製」からの作成を推奨)
- プロジェクトを作成される際、追加で付与いただくラベルの設定を実施
- プロジェクトの作成後、プロジェクトIDを確認
- Platformにログイン
- 「ジョブ定義」よりNotebookを起動
- NotebookのTerminalを起動し、以下のGitリポジトリからClone　

<code>git clone https://github.com/abeja-inc/abeja-platform-samples.git</code>

- 「Notebook」→「annotation_project_copy_for_segmentation」配下の「secret.yaml」を開く
- 「secret.yaml」の各認証情報を入力し、保存 (コピー先のプロジェクトIDもここで入力)
- コピーを実施したいプロジェクトのJSONファイルをNotebookへコピー
- 「annotation-prj-copy_for_segmentation.ipynb」を開く
- Notebookファイル上の以下の項目をアップロードしたJSONファイル名に編集

<code>ANNOTATION_JSON_NAME = 'XXXXXXXXXXXX.json'</code>

- Notebookを上部から実行「Run」
