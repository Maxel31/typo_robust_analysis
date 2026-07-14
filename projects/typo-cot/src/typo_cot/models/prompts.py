"""ベンチマーク別CoTプロンプトテンプレート.

各ベンチマークに対応したCoT推論プロンプトを提供する。
- GSM8K: 8-Shot CoT
- MMLU: 5-Shot CoT
- MMLU-Pro: 5-Shot CoT
- SQuAD v2: CoTなし（読解QAタスク）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PromptResult:
    """プロンプト生成結果.

    Attributes:
        system_prompt: システムプロンプト（Few-shot例示を含む）
        user_prompt: ユーザープロンプト（質問）
        question_text: 分析対象の質問文テキスト（摂動適用対象）
        question_start_in_full: 完全なプロンプト内での質問文開始位置（文字単位）
        question_end_in_full: 完全なプロンプト内での質問文終了位置（文字単位）
        question_with_choices_end: 質問文＋選択肢の終了位置（AttnLRP分析用）
        context_start_in_full: コンテキストの開始位置（SQuAD v2の場合）
        context_end_in_full: コンテキストの終了位置（SQuAD v2の場合）
        subject: サブジェクト名（MMLU/MMLU-Proの場合）
        answer_start: 参照箇所の開始位置（SQuAD v2の場合）
        answer_end: 参照箇所の終了位置（SQuAD v2の場合）
    """

    system_prompt: str
    user_prompt: str
    question_text: str = ""
    question_start_in_full: int = 0
    question_end_in_full: int = 0
    question_with_choices_end: int = 0  # 選択肢を含む終了位置
    context_start_in_full: int | None = None  # SQuAD用: コンテキスト開始位置
    context_end_in_full: int | None = None  # SQuAD用: コンテキスト終了位置
    subject: str | None = None
    answer_start: int | None = None
    answer_end: int | None = None

    def get_full_prompt(self) -> str:
        """完全なプロンプトを取得."""
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{self.user_prompt}"
        return self.user_prompt


class BasePromptTemplate(ABC):
    """プロンプトテンプレートの基底クラス."""

    @abstractmethod
    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """プロンプトを生成する.

        Args:
            question: 質問文
            choices: 選択肢（選択式の場合）
            context: コンテキスト（読解問題の場合）
            subject: サブジェクト名（MMLU/MMLU-Proの場合）
            answer_start: 回答の開始位置（SQuAD v2の場合）

        Returns:
            生成されたプロンプト
        """
        pass

    @abstractmethod
    def get_default_num_shots(self) -> int:
        """デフォルトのショット数を取得."""
        pass


class GSM8KPromptTemplate(BasePromptTemplate):
    """GSM8K用8-Shot CoTプロンプトテンプレート.

    数学問題に対応し、"The answer is [数値]." 形式で出力させる。
    各例示は明確に区切られ、モデルが混乱しないようにする。
    """

    SYSTEM_INSTRUCTION = """Solve the following math problems step by step. Show your reasoning, then provide the final answer in the format "The answer is [number]."

Here are some examples:"""

    # GSM8Kの8-Shot例示（公式のトレーニングセットから選択）
    FEW_SHOT_EXAMPLES = [
        {
            "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
            "reasoning": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.",
            "answer": "11",
        },
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.",
            "answer": "6",
        },
        {
            "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
            "answer": "5",
        },
        {
            "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
            "answer": "39",
        },
        {
            "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.",
            "answer": "8",
        },
        {
            "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "reasoning": "Shawn started with 5 toys. He got 2 toys from his mom and 2 toys from his dad. 2 + 2 = 4 new toys. 5 + 4 = 9 toys total.",
            "answer": "9",
        },
        {
            "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "reasoning": "There were originally 9 computers. For each day from monday to thursday, 5 more computers were added. That is 4 days. So 5 * 4 = 20 computers were added. 9 + 20 = 29.",
            "answer": "29",
        },
        {
            "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more on wednesday, he had 35 - 2 = 33 golf balls.",
            "answer": "33",
        },
    ]

    def get_default_num_shots(self) -> int:
        """デフォルトのショット数を取得."""
        return 8

    def _format_example(self, example: dict, index: int) -> str:
        """1つの例示をフォーマット."""
        return f"""
---
Example {index + 1}:
Problem: {example["question"]}

Solution: {example["reasoning"]}
The answer is {example["answer"]}.
---"""

    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """GSM8Kプロンプトを生成."""
        # システムプロンプトを構築
        system_parts = [self.SYSTEM_INSTRUCTION]

        # Few-shot例示を追加（明確な区切り付き）
        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES):
            system_parts.append(self._format_example(ex, i))

        system_parts.append("\nNow solve the following problem:\n")
        system_prompt = "".join(system_parts)

        # ユーザープロンプト
        user_prompt = f"""Problem: {question}

Solution:"""

        # 質問文範囲を計算（完全プロンプト内での位置）
        question_prefix = "Problem: "
        question_start_in_user = len(question_prefix)
        question_end_in_user = question_start_in_user + len(question)

        # full_prompt内での位置（system_prompt + "\n\n" の分を加算）
        prefix_length = len(system_prompt) + 2  # "\n\n"の長さ
        question_start_in_full = prefix_length + question_start_in_user
        question_end_in_full = prefix_length + question_end_in_user

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_text=question,
            question_start_in_full=question_start_in_full,
            question_end_in_full=question_end_in_full,
            question_with_choices_end=question_end_in_full,  # GSM8Kは選択肢なし
        )


class MMLUPromptTemplate(BasePromptTemplate):
    """MMLU用5-Shot CoTプロンプトテンプレート.

    選択式問題に対応し、"The answer is (X)" 形式で出力させる。
    各例示は明確に区切られ、モデルが混乱しないようにする。
    """

    SYSTEM_INSTRUCTION = """The following are multiple-choice questions about {subject}. For each question, think through the problem step by step, then provide your final answer in the format "The answer is (X)" where X is the correct letter choice.

Here are some examples:"""

    # MMLUの5-Shot例示（一般的な例）
    FEW_SHOT_EXAMPLES = [
        {
            "question": "What is the capital of France?",
            "options": ["London", "Paris", "Berlin", "Madrid"],
            "reasoning": "France is a country in Western Europe. Paris is both the capital and the largest city of France, known for landmarks like the Eiffel Tower.",
            "answer": "B",
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Saturn"],
            "reasoning": "The Red Planet is a nickname given due to the reddish appearance caused by iron oxide (rust) on its surface. This planet is Mars.",
            "answer": "B",
        },
        {
            "question": "What is 15% of 200?",
            "options": ["25", "30", "35", "40"],
            "reasoning": "To find 15% of 200, I calculate: 15/100 × 200 = 0.15 × 200 = 30.",
            "answer": "B",
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "options": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
            "reasoning": "'Romeo and Juliet' is a famous tragedy written in the late 16th century by the English playwright William Shakespeare.",
            "answer": "B",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "options": ["Ag", "Au", "Fe", "Cu"],
            "reasoning": "The chemical symbol for gold comes from the Latin word 'aurum'. Therefore, gold's symbol is Au. (Ag is silver, Fe is iron, Cu is copper.)",
            "answer": "B",
        },
    ]

    def get_default_num_shots(self) -> int:
        """デフォルトのショット数を取得."""
        return 5

    def _format_options(self, choices: list[str]) -> str:
        """選択肢をフォーマット（空白区切り）."""
        letters = "ABCDEFGHIJ"
        return " ".join(f"({letters[i]}) {choice}" for i, choice in enumerate(choices))

    def _format_example(self, example: dict, index: int) -> str:
        """1つの例示をフォーマット."""
        options_str = self._format_options(example["options"])
        return f"""
---
Example {index + 1}:
Question: {example["question"]}
{options_str}

Step-by-step reasoning: {example["reasoning"]}
The answer is ({example["answer"]}).
---"""

    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """MMLUプロンプトを生成.

        Args:
            question: 質問文（または質問文＋選択肢が含まれたテキスト）
            choices: 選択肢リスト（Noneまたは空の場合、questionに選択肢が含まれると仮定）
            context: 未使用
            subject: サブジェクト名
            answer_start: 未使用
        """
        # サブジェクト名を整形
        subject_name = subject.replace("_", " ") if subject else "general knowledge"

        # システムプロンプトを構築
        system_parts = [self.SYSTEM_INSTRUCTION.format(subject=subject_name)]

        # Few-shot例示を追加（明確な区切り付き）
        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES):
            system_parts.append(self._format_example(ex, i))

        system_parts.append("\nNow answer the following question:\n")
        system_prompt = "".join(system_parts)

        # ユーザープロンプト
        # choicesが空またはNoneの場合、questionに選択肢が含まれていると仮定
        if choices:
            options_str = self._format_options(choices)
            question_with_options = f"{question}\n{options_str}"
        else:
            # questionにすでに選択肢が含まれている（摂動データの場合）
            question_with_options = question

        user_prompt = f"""Question: {question_with_options}

Step-by-step reasoning:"""

        # 質問文範囲を計算（完全プロンプト内での位置）
        question_prefix = "Question: "
        question_start_in_user = len(question_prefix)

        # choicesがある場合は質問文のみ、ない場合は全体がquestion
        if choices:
            question_end_in_user = question_start_in_user + len(question)
        else:
            # 選択肢込みの場合、改行の前までを質問文とする（概算）
            first_newline = question.find("\n")
            if first_newline > 0:
                question_end_in_user = question_start_in_user + first_newline
            else:
                question_end_in_user = question_start_in_user + len(question)

        # full_prompt内での位置（system_prompt + "\n\n" の分を加算）
        prefix_length = len(system_prompt) + 2  # "\n\n"の長さ
        question_start_in_full = prefix_length + question_start_in_user
        question_end_in_full = prefix_length + question_end_in_user

        # 質問文＋選択肢の終了位置（"\n\nStep-by-step reasoning:"の直前まで）
        question_with_choices_end_in_user = len(user_prompt) - len("\n\nStep-by-step reasoning:")
        question_with_choices_end = prefix_length + question_with_choices_end_in_user

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_text=question,
            question_start_in_full=question_start_in_full,
            question_end_in_full=question_end_in_full,
            question_with_choices_end=question_with_choices_end,
            subject=subject,
        )


class MMLUProPromptTemplate(BasePromptTemplate):
    """MMLU-Pro用5-Shot CoTプロンプトテンプレート.

    選択式問題に対応し、"The answer is (X)" 形式で出力させる。
    MMLU-Proは10個の選択肢を持つことがある。
    各例示は明確に区切られ、モデルが混乱しないようにする。
    """

    SYSTEM_INSTRUCTION = """The following are multiple-choice questions about {subject}. For each question, think through the problem step by step, then provide your final answer in the format "The answer is (X)" where X is the correct letter choice.

Here are some examples:"""

    # MMLU-Proの5-Shot例示
    FEW_SHOT_EXAMPLES = [
        {
            "question": "What is the primary function of mitochondria in a cell?",
            "options": [
                "Protein synthesis",
                "Energy production (ATP)",
                "Cell division",
                "Waste removal",
                "DNA replication",
            ],
            "reasoning": "Mitochondria are known as the powerhouses of the cell. They are responsible for producing ATP (adenosine triphosphate) through cellular respiration.",
            "answer": "B",
        },
        {
            "question": "In economics, what does GDP stand for?",
            "options": [
                "General Domestic Product",
                "Gross Domestic Product",
                "Global Development Program",
                "Government Debt Position",
                "Growth and Development Policy",
            ],
            "reasoning": "GDP is a common economic indicator that measures the total value of goods and services produced within a country. It stands for Gross Domestic Product.",
            "answer": "B",
        },
        {
            "question": "Which programming paradigm emphasizes immutability and pure functions?",
            "options": [
                "Object-oriented programming",
                "Procedural programming",
                "Functional programming",
                "Logic programming",
                "Event-driven programming",
            ],
            "reasoning": "The programming paradigm that emphasizes immutability (data that cannot be changed) and pure functions (functions without side effects) is functional programming.",
            "answer": "C",
        },
        {
            "question": "What is the time complexity of binary search?",
            "options": [
                "O(1)",
                "O(n)",
                "O(log n)",
                "O(n log n)",
                "O(n²)",
            ],
            "reasoning": "Binary search works by repeatedly dividing the search space in half. Each comparison eliminates half of the remaining elements. This results in a time complexity of O(log n).",
            "answer": "C",
        },
        {
            "question": "Which law of thermodynamics states that entropy always increases?",
            "options": [
                "Zeroth law",
                "First law",
                "Second law",
                "Third law",
                "Fourth law",
            ],
            "reasoning": "The second law of thermodynamics states that the total entropy of an isolated system always increases over time. This is also related to the direction of natural processes.",
            "answer": "C",
        },
    ]

    def get_default_num_shots(self) -> int:
        """デフォルトのショット数を取得."""
        return 5

    def _format_options(self, choices: list[str]) -> str:
        """選択肢をフォーマット（空白区切り）."""
        letters = "ABCDEFGHIJ"
        return " ".join(f"({letters[i]}) {choice}" for i, choice in enumerate(choices))

    def _format_example(self, example: dict, index: int) -> str:
        """1つの例示をフォーマット."""
        options_str = self._format_options(example["options"])
        return f"""
---
Example {index + 1}:
Question: {example["question"]}
{options_str}

Step-by-step reasoning: {example["reasoning"]}
The answer is ({example["answer"]}).
---"""

    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """MMLU-Proプロンプトを生成.

        Args:
            question: 質問文（または質問文＋選択肢が含まれたテキスト）
            choices: 選択肢リスト（Noneまたは空の場合、questionに選択肢が含まれると仮定）
            context: 未使用
            subject: サブジェクト名
            answer_start: 未使用
        """
        # サブジェクト名を整形
        subject_name = subject.replace("_", " ") if subject else "general knowledge"

        # システムプロンプトを構築
        system_parts = [self.SYSTEM_INSTRUCTION.format(subject=subject_name)]

        # Few-shot例示を追加（明確な区切り付き）
        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES):
            system_parts.append(self._format_example(ex, i))

        system_parts.append("\nNow answer the following question:\n")
        system_prompt = "".join(system_parts)

        # ユーザープロンプト
        # choicesが空またはNoneの場合、questionに選択肢が含まれていると仮定
        if choices:
            options_str = self._format_options(choices)
            question_with_options = f"{question}\n{options_str}"
        else:
            # questionにすでに選択肢が含まれている（摂動データの場合）
            question_with_options = question

        user_prompt = f"""Question: {question_with_options}

Step-by-step reasoning:"""

        # 質問文範囲を計算（完全プロンプト内での位置）
        question_prefix = "Question: "
        question_start_in_user = len(question_prefix)

        # choicesがある場合は質問文のみ、ない場合は全体がquestion
        if choices:
            question_end_in_user = question_start_in_user + len(question)
        else:
            # 選択肢込みの場合、改行の前までを質問文とする（概算）
            first_newline = question.find("\n")
            if first_newline > 0:
                question_end_in_user = question_start_in_user + first_newline
            else:
                question_end_in_user = question_start_in_user + len(question)

        # full_prompt内での位置（system_prompt + "\n\n" の分を加算）
        prefix_length = len(system_prompt) + 2  # "\n\n"の長さ
        question_start_in_full = prefix_length + question_start_in_user
        question_end_in_full = prefix_length + question_end_in_user

        # 質問文＋選択肢の終了位置（"\n\nStep-by-step reasoning:"の直前まで）
        question_with_choices_end_in_user = len(user_prompt) - len("\n\nStep-by-step reasoning:")
        question_with_choices_end = prefix_length + question_with_choices_end_in_user

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_text=question,
            question_start_in_full=question_start_in_full,
            question_end_in_full=question_end_in_full,
            question_with_choices_end=question_with_choices_end,
            subject=subject,
        )


class SQuADv2PromptTemplate(BasePromptTemplate):
    """SQuAD v2用プロンプトテンプレート.

    読解QAタスクのため、CoT推論は行わない。
    参照箇所のアノテーション情報を保存する。
    """

    def get_default_num_shots(self) -> int:
        """SQuAD v2はショットなし."""
        return 0

    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """SQuAD v2プロンプトを生成."""
        if context is None:
            raise ValueError("SQuAD問題にはコンテキストが必要です")

        # シンプルな読解QA形式
        system_prompt = ""
        user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        # コンテキスト範囲を計算
        # user_prompt = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        context_prefix = "Context: "
        context_start_in_full = len(context_prefix)
        context_end_in_full = context_start_in_full + len(context)

        # 質問文範囲を計算（system_promptが空なのでuser_prompt内の位置がそのまま使える）
        question_prefix = f"Context: {context}\n\nQuestion: "
        question_start_in_full = len(question_prefix)
        question_end_in_full = question_start_in_full + len(question)

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_text=question,
            question_start_in_full=question_start_in_full,
            question_end_in_full=question_end_in_full,
            question_with_choices_end=question_end_in_full,  # SQuADは選択肢なし
            context_start_in_full=context_start_in_full,
            context_end_in_full=context_end_in_full,
            answer_start=answer_start,
        )


class ARCPromptTemplate(MMLUPromptTemplate):
    """ARC (AI2 Reasoning Challenge) 用5-Shot CoTプロンプトテンプレート.

    科学的推論の4択問題に対応。MMLUPromptTemplateを継承し、
    指示文とfew-shot例を科学分野向けにカスタマイズ。
    """

    SYSTEM_INSTRUCTION = """The following are multiple-choice science questions. For each question, think through the problem step by step, then provide your final answer in the format "The answer is (X)" where X is the correct letter choice.

Here are some examples:"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "Which property of a mineral can be determined just by looking at it?",
            "options": ["luster", "hardness", "weight", "streak"],
            "reasoning": "Luster describes how light reflects off the surface of a mineral. This can be observed just by looking at it. Hardness requires a scratch test, weight requires a scale, and streak requires rubbing the mineral on a porcelain plate.",
            "answer": "A",
        },
        {
            "question": "A student is trying to identify a mineral that has a hardness of ite and can scratch glass. Which mineral could it be?",
            "options": ["calcite", "fluorite", "quartz", "gypsum"],
            "reasoning": "Glass has a hardness of about 5.5 on the Mohs scale. Quartz has a hardness of 7 and can scratch glass. Calcite (3), fluorite (4), and gypsum (2) are all softer than glass.",
            "answer": "C",
        },
        {
            "question": "What is the main function of the roots of a plant?",
            "options": [
                "to absorb water and nutrients from the soil",
                "to make food for the plant",
                "to attract insects for pollination",
                "to release oxygen into the air",
            ],
            "reasoning": "Roots primarily absorb water and minerals from the soil. Leaves make food through photosynthesis, flowers attract pollinators, and leaves release oxygen.",
            "answer": "A",
        },
        {
            "question": "Which of the following is an example of a chemical change?",
            "options": [
                "ice melting into water",
                "wood burning in a fire",
                "salt dissolving in water",
                "cutting paper with scissors",
            ],
            "reasoning": "A chemical change creates new substances. Burning wood produces ash, carbon dioxide, and water vapor - all new substances. Melting, dissolving, and cutting are physical changes.",
            "answer": "B",
        },
        {
            "question": "A ball is thrown straight up into the air. What happens to the speed of the ball as it rises?",
            "options": [
                "It increases",
                "It decreases",
                "It remains the same",
                "It first increases then decreases",
            ],
            "reasoning": "When a ball is thrown upward, gravity acts on it in the downward direction, decelerating it. So the speed decreases as it rises, reaching zero at the highest point.",
            "answer": "B",
        },
    ]

    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """ARCプロンプトを生成（subject固定で親クラスに委譲）."""
        return super().generate(
            question=question,
            choices=choices,
            context=context,
            subject="science",
            answer_start=answer_start,
        )


class CommonsenseQAPromptTemplate(MMLUPromptTemplate):
    """CommonsenseQA用5-Shot CoTプロンプトテンプレート.

    常識推論の5択問題に対応。MMLUPromptTemplateを継承し、
    指示文とfew-shot例を常識推論向けにカスタマイズ。
    """

    SYSTEM_INSTRUCTION = """The following are multiple-choice commonsense reasoning questions. For each question, think through the problem step by step, then provide your final answer in the format "The answer is (X)" where X is the correct letter choice.

Here are some examples:"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "Where would you find a dog on a leash?",
            "options": ["dog show", "move forward", "park", "school", "vet"],
            "reasoning": "A dog on a leash is typically being walked by its owner. Parks are common places where people walk their dogs on leashes.",
            "answer": "C",
        },
        {
            "question": "What do people do when they don't understand something?",
            "options": ["educate", "communicate", "ask questions", "complete", "argue"],
            "reasoning": "When people don't understand something, the natural response is to seek clarification by asking questions to gain better understanding.",
            "answer": "C",
        },
        {
            "question": "If you want to set a romantic atmosphere, what might you light?",
            "options": ["fire", "candle", "torch", "lamp", "match"],
            "reasoning": "Candles are commonly associated with creating a romantic atmosphere due to their soft, warm glow. While a match can light things, it's not what you keep lit for ambiance.",
            "answer": "B",
        },
        {
            "question": "Where would you put a used tissue?",
            "options": ["pocket", "trash can", "box", "ground", "table"],
            "reasoning": "A used tissue is waste and should be disposed of properly. A trash can is the appropriate place to put used tissues.",
            "answer": "B",
        },
        {
            "question": "What could happen to a house if it is not maintained?",
            "options": [
                "fall down",
                "get new roof",
                "become cleaner",
                "increase in value",
                "get painted",
            ],
            "reasoning": "Without maintenance, a house deteriorates over time. Structural elements weaken, and eventually the house could fall down or collapse.",
            "answer": "A",
        },
    ]

    def generate(
        self,
        question: str,
        choices: list[str] | None = None,
        context: str | None = None,
        subject: str | None = None,
        answer_start: int | None = None,
    ) -> PromptResult:
        """CommonsenseQAプロンプトを生成（subject固定で親クラスに委譲）."""
        return super().generate(
            question=question,
            choices=choices,
            context=context,
            subject="commonsense reasoning",
            answer_start=answer_start,
        )


class BBHPromptTemplate(GSM8KPromptTemplate):
    """BIG-Bench Hard (BBH) 用 3-Shot CoT プロンプト.

    BBH は 23 サブタスクそれぞれで質問構造が異なるが、いずれも
    "Q: ... A: ... So the answer is X" 形式の標準 CoT プロンプトに従う.
    本実装ではタスク非依存の少数 few-shot で汎用 CoT を促す.
    """

    SYSTEM_INSTRUCTION = """Solve the following challenging reasoning problem step by step. After your reasoning, give the final answer in the format "The answer is X" where X is the requested answer.

Here are some examples:"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "If you follow these instructions, do you return to the starting point? Turn left. Turn around. Turn left. Take 7 steps. Take 2 steps. Take 4 steps. Take 8 steps.",
            "reasoning": "We start facing forward. Turn left, then turn around: facing right. Turn left: facing forward. Take 7+2+4+8 = 21 steps forward. We are 21 steps from origin, not back.",
            "answer": "No",
        },
        {
            "question": "Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?",
            "reasoning": "Today is December 24, 1937. 10 days before December 24 is December 14, 1937.",
            "answer": "12/14/1937",
        },
        {
            "question": "Sort the following words alphabetically: List: oven cogent ash burnish.",
            "reasoning": "Alphabetic order: ash, burnish, cogent, oven.",
            "answer": "ash burnish cogent oven",
        },
    ]

    def get_default_num_shots(self) -> int:
        return 3


class MATHPromptTemplate(GSM8KPromptTemplate):
    """MATH (Hendrycks) 用 4-Shot CoT プロンプト.

    回答は \\boxed{...} 形式の LaTeX 表現で出力させる. 既存の
    GSM8KPromptTemplate を継承し、SYSTEM_INSTRUCTION と FEW_SHOT を
    MATH 向けに差し替える.
    """

    SYSTEM_INSTRUCTION = """Solve the following math problems step by step. Show your reasoning, then provide the final answer enclosed in \\boxed{} and also as "The answer is \\boxed{...}".

Here are some examples:"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "What is the value of $\\dfrac{3 \\times 4}{6}$?",
            "reasoning": "$3 \\times 4 = 12$, and $12/6 = 2$.",
            "answer": "\\boxed{2}",
        },
        {
            "question": "Solve for $x$: $2x + 3 = 11$.",
            "reasoning": "Subtract 3 from both sides: $2x = 8$. Divide by 2: $x = 4$.",
            "answer": "\\boxed{4}",
        },
        {
            "question": "What is the slope of the line passing through $(1,2)$ and $(3,6)$?",
            "reasoning": "Slope = $(6-2)/(3-1) = 4/2 = 2$.",
            "answer": "\\boxed{2}",
        },
        {
            "question": "What is $7! / 5!$?",
            "reasoning": "$7!/5! = 7 \\times 6 = 42$.",
            "answer": "\\boxed{42}",
        },
    ]

    def get_default_num_shots(self) -> int:
        return 4


class StrategyQAPromptTemplate(GSM8KPromptTemplate):
    """StrategyQA 用 6-Shot CoT プロンプト.

    多段階常識推論. 質問は1文 yes/no で、回答は "The answer is Yes/No".
    """

    SYSTEM_INSTRUCTION = """Answer the following yes/no questions that require multi-step reasoning. Think through the problem step by step, then provide the final answer as "The answer is Yes" or "The answer is No".

Here are some examples:"""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "Do hamsters provide food for any animals?",
            "reasoning": "Hamsters are small mammals. Predators such as snakes, hawks, and foxes prey on hamsters in the wild. Therefore hamsters provide food for predator animals.",
            "answer": "Yes",
        },
        {
            "question": "Could Brooke Shields succeed at University of Pennsylvania?",
            "reasoning": "Brooke Shields graduated from Princeton University, which is also an Ivy League school. She is intellectually capable of succeeding at a comparable institution like the University of Pennsylvania.",
            "answer": "Yes",
        },
        {
            "question": "Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
            "reasoning": "Hydrogen has atomic number 1. 1 squared = 1. There were 5 Spice Girls. 1 does not exceed 5.",
            "answer": "No",
        },
        {
            "question": "Yes or no: Is it common to see frost during some college commencements?",
            "reasoning": "Some colleges have winter commencements in December. Frost commonly occurs in December in the northern hemisphere. So yes, frost is sometimes seen during commencements.",
            "answer": "Yes",
        },
        {
            "question": "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
            "reasoning": "Llamas have a gestation period of about 11 months. The 1945-46 war lasted about a year, which is too short for two consecutive births.",
            "answer": "No",
        },
        {
            "question": "Yes or no: Would a pear sink in water?",
            "reasoning": "Pears have a density less than water (about 0.6 g/cm^3). Objects with density less than water float. So pears do not sink.",
            "answer": "No",
        },
    ]

    def get_default_num_shots(self) -> int:
        return 6


def create_prompt_template(benchmark: str) -> BasePromptTemplate:
    """ベンチマーク名からプロンプトテンプレートを作成するファクトリ関数.

    Args:
        benchmark: ベンチマーク名

    Returns:
        対応するプロンプトテンプレート

    Raises:
        ValueError: 不明なベンチマーク名の場合
    """
    templates = {
        "mmlu": MMLUPromptTemplate,
        "mmlu_pro": MMLUProPromptTemplate,
        "gsm8k": GSM8KPromptTemplate,
        "squad_v2": SQuADv2PromptTemplate,
        "arc": ARCPromptTemplate,
        "commonsense_qa": CommonsenseQAPromptTemplate,
        "bbh": BBHPromptTemplate,
        "math": MATHPromptTemplate,
        "strategy_qa": StrategyQAPromptTemplate,
    }

    if benchmark not in templates:
        raise ValueError(f"不明なベンチマーク: {benchmark}. 利用可能: {list(templates.keys())}")

    return templates[benchmark]()
