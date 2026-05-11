# ruff: noqa: E501

from langchain_core.prompts import ChatPromptTemplate

ANSWER_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator assessing the quality of answers. "
            "Your task is to compare a generated answer against a reference answer. "
            "You MUST respond strictly with a JSON object. "
            "Do not include any conversational text, preambles, or markdown blocks like ```json. "
            "\n{format_instructions}",
        ),
        (
            "user",
            """Evaluate the following:
        
Question:
{question}

Generated Answer:
{generated_answer}

Reference Answer:
{reference_answer}

Evaluation Criteria:
1. Accuracy: How factually correct is it compared to the reference answer? (1-5, only 5 for perfect matches, 1 if wrong).
2. Completeness: Does it cover all information from the reference answer? (1-5).
3. Relevance: Does it directly answer the question without unnecessary fluff? (1-5).

Provide your scores and detailed reasoning for each dimension strictly in the specified JSON format.""",
        ),
    ]
)
