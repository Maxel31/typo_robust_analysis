"""実験7: 校正器ラダー (defense) パッケージ.

弱・中・強の3段の校正器 (pyspellchecker / ニューラル seq2seq / LLM) を
共通インターフェース correct(text) -> text でラップし、
校正後テキストの語単位復元判定と評価集計を提供する。
"""
