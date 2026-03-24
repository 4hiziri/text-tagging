# text-tagging

spaCy と scikit-learn を使って、日本語テキストから TF-IDF ベースで重要キーワードを抽出するサンプルです。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download ja_core_news_sm
```

## 使い方

### 1) 引数でテキストを渡す

```bash
python tag_text.py "今日は自然言語処理の勉強として、spaCyとTF-IDFを使ったキーワード抽出を試しています。"
```

### 2) 標準入力から渡す

```bash
echo "機械学習を使ってテキスト分類の精度を改善する方法を調査しています。" | python tag_text.py
```

デフォルトで上位5件を表示します。件数を変更したい場合は `--top-n` を使ってください。

```bash
python tag_text.py "日本語テキスト解析を行います。" --top-n 3
```

## 実装メモ

- spaCy で形態素解析した結果から不要語を除外します。
- テキストを文単位で分割し、各文を1ドキュメントとして TF-IDF を計算します。
- 各語の文ごとの TF-IDF スコアの最大値を採用し、上位キーワードを表示します。
