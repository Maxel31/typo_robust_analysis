# datasets

複数プロジェクトで共有するデータセットの置き場。

- 大容量データは git 管理せず、ここに配置（または外部ストレージへ symlink）する。
- コードからは `typo_utils.data.loaders.resolve_dataset("<name>")` で
  `datasets/<name>` を解決する。
- 既存の共有データ（例: `kanolab/datasets`）を使う場合は symlink を張る:

  ```bash
  ln -s /path/to/kanolab/datasets/STaMP datasets/STaMP
  ```
