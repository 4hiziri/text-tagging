#!/usr/bin/env python3
"""spaCy + scikit-learn による日本語テキストのタグ付けサンプル。"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any, Iterable, List


STOP_POS = {
    "PUNCT",
    "SYM",
    "SPACE",
    "AUX",
    "ADP",
    "PART",
    "CCONJ",
    "SCONJ",
    "DET",
    "PRON",
}


def load_nlp(model_name: str = "ja_core_news_lg"):
    try:
        import spacy

        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            "spaCy の日本語モデルが見つかりません。\n"
            "次のコマンドでインストールしてください: python -m spacy download ja_core_news_lg"
        ) from exc
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "spaCy がインストールされていません。\n"
            "次のコマンドでインストールしてください: pip install -r requirements.txt"
        ) from exc


def tokenize_for_keywords(doc: Any) -> List[str]:
    tokens: List[str] = []

    for token in doc:
        lemma = token.lemma_.strip()
        if not lemma:
            continue
        if token.is_stop:
            continue
        if token.pos_ in STOP_POS:
            continue
        if token.is_punct or token.is_space:
            continue
        if len(lemma) <= 1:
            continue
        tokens.append(lemma)

    return tokens


def build_corpus_from_sentences(doc: Any) -> List[str]:
    corpus: List[str] = []
    for sent in doc.sents:
        sentence_tokens = tokenize_for_keywords(sent)
        if sentence_tokens:
            corpus.append(" ".join(sentence_tokens))

    if corpus:
        return corpus

    # 文区切りが取れない/有効語が少ない場合のフォールバック
    fallback_tokens = tokenize_for_keywords(doc)
    if fallback_tokens:
        return [" ".join(fallback_tokens)]
    return []


def extract_top_keywords_from_corpus(
    corpus: Iterable[str], top_n: int = 5
) -> List[str]:
    corpus_list = [item for item in corpus if item.strip()]
    if not corpus_list:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn がインストールされていません。\n"
            "次のコマンドでインストールしてください: pip install -r requirements.txt"
        ) from exc

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", norm="l2", sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(corpus_list)

    feature_names = vectorizer.get_feature_names_out()
    term_scores = defaultdict(float)

    for row in matrix.toarray():
        for i, score in enumerate(row):
            if score > term_scores[feature_names[i]]:
                term_scores[feature_names[i]] = float(score)

    ranked = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:top_n]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="spaCy で日本語を解析し、TF-IDF で重要キーワードを抽出します。"
    )
    parser.add_argument("text", nargs="?", help="解析したい日本語テキスト")
    parser.add_argument(
        "--top-n", type=int, default=5, help="抽出するキーワード数 (デフォルト: 5)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else sys.stdin.read().strip()

    if not text:
        raise SystemExit(
            "テキストが空です。引数または標準入力で日本語テキストを指定してください。"
        )
    if args.top_n <= 0:
        raise SystemExit("--top-n には 1 以上の整数を指定してください。")

    nlp = load_nlp()
    doc = nlp(text)
    corpus = build_corpus_from_sentences(doc)
    keywords = extract_top_keywords_from_corpus(corpus, top_n=args.top_n)

    print("=== 抽出キーワード ===")
    if not keywords:
        print("キーワードを抽出できませんでした。")
        return

    for i, keyword in enumerate(keywords, start=1):
        print(f"{i}. {keyword}")


if __name__ == "__main__":
    main()
