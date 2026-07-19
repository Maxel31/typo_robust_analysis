"""実験1/3 用のサンプル横断レコード定義.

データアクセスを薄い層に隔離するための中間表現。現時点ではアーカイブの
baseline/perturbed results.json から構築するが、Step 0 の master table
(sample_id / model / benchmark / condition / question_text / cot_text /
answer_span / answer_pred / ... スキーマ) が完成したら、そこから1行で
構築できるようにフィールドを揃えてある。
"""

from dataclasses import dataclass, field


@dataclass
class PairRecord:
    """clean × 摂動 (LXT-4 / Random-4) の1サンプル対.

    Attributes:
        sample_id: サンプル ID (アーカイブ/master table 共通)
        model: HuggingFace モデル名
        benchmark: ベンチマーク名 (gsm8k / mmlu / ...)
        question_clean: clean 質問文 (baseline record の question)
        question_typo: typo 質問文 (perturbed record の question。MMLU 系は
            選択肢込みテキストで choices_typo=None)
        choices_clean: clean 側の選択肢リスト (自由記述は None)
        choices_typo: typo 側の選択肢リスト (摂動データは通常 None=質問に内包)
        subset: サブセット名 (プロンプトの subject に使用)
        correct_answer: 正解
        cot_clean: clean 条件の生成テキスト (答え句込みの CoT)
        cot_typo: typo 条件の生成テキスト
        answer_clean: clean 条件でアーカイブが抽出した答え
        answer_typo: typo 条件でアーカイブが抽出した答え
        is_correct_clean: clean 条件で正解だったか (主推定量の条件付けに使用)
        extra: 付加情報 (R_Q/R_C 等、実験3 の precision@k 用)
    """

    sample_id: str
    model: str
    benchmark: str
    question_clean: str
    question_typo: str
    choices_clean: list[str] | None
    choices_typo: list[str] | None
    subset: str | None
    correct_answer: str
    cot_clean: str
    cot_typo: str
    answer_clean: str
    answer_typo: str
    is_correct_clean: bool
    extra: dict = field(default_factory=dict)
