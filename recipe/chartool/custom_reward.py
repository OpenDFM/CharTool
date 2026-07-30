import logging
import re
import sys
import os
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logger = logging.getLogger(__name__)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("LLM_AS_A_JUDGE_BASE"),
)


def get_first_available_model() -> str | None:
    """Return the first available model ID, or None if not accessible."""
    try:
        models = client.models.list()
        if models.data:
            model_name = models.data[0].id
            logger.info(f"Using model '{model_name}' from {client.base_url} for reward scoring.")
            return model_name
        logger.error("No models found for reward scoring.")
    except Exception as e:
        logger.error(f"Failed to get model from {client.base_url}: {e}")
    return None

model_name = get_first_available_model()

def compute_score_chart(
    data_source: str, solution_str: str, ground_truth: str, extra_info=None
) -> float:
    """
    Compute reward score for model solutions with robust handling of various formats.

    Returns a weighted combination of:
    - Accuracy reward (1.0 weight): Whether the answer is semantically correct
    - Format reward (0.1 weight): Whether the output follows expected format
    - Tool reward (0.2 weight): Whether tools were used when answer is correct
    """

    is_format_error = False

    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    answer_text = ""

    predict_no_think = (
        solution_str.split("</think>")[-1].strip()
        if "</think>" in solution_str
        else solution_str.strip()
    )

    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True
    
    count_tool_1 = solution_str.count("<tool_call>")
    count_tool_2 = solution_str.count("</tool_call>")
    if count_tool_1 != count_tool_2:
        is_format_error = True

    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        is_format_error = True

    answer_text = answer_text.strip()


    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()  # Use full text as last resort

    if not answer_text:
        acc_reward = -1.0
    elif len(answer_text) >= 1000: # Penalize excessively long answers (potential judge hacking)
        acc_reward = -1.0
        is_format_error = True
    else:
        question_text = extra_info.get("question", "") if extra_info else ""
        acc_reward = llm_judge(question_text, ground_truth, answer_text)        

    has_tool_usage = False

    tool_call_matches = re.finditer(
        r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL
    )
    for match in tool_call_matches:
        tool_call_content = match.group(1).strip()
        code_match = re.search(r'"code"\s*:\s*"([^"]*)"', tool_call_content)
        if code_match:
            code_content = code_match.group(1).strip()
            if code_content and code_content not in ["", "\\n", "\\t", "\\r\\n"]:
                has_tool_usage = True
                break

    if not has_tool_usage:
        tool_response_matches = re.finditer(
            r"<tool_response>(.*?)</tool_response>", solution_str, re.DOTALL
        )
        for match in tool_response_matches:
            response_content = match.group(1).strip()
            if response_content:
                has_tool_usage = True
                break

    tool_reward = 1.0 if has_tool_usage and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 1.0

    if is_format_error or not answer_text:
        logger.debug(
            f"Format issue detected:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Format error: {is_format_error}\n"
            f"Tool usage: {has_tool_usage}"
        )

    final_score = acc_reward + 0.1 * format_reward + 0.2 * tool_reward

    acc = 1.0 if acc_reward > 0.5 else 0.0
    format_acc = 0.0 if is_format_error else 1.0
    reward_dict = {
        "score": final_score,
        "acc": acc,
        "format_acc": format_acc,
    }

    return reward_dict


CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""


def llm_judge(question_text: str, ground_truth: str, answer_text: str) -> float:
    if not client or not model_name:
        logger.warning(
            "Reward function client not initialized or model name not found."
        )
        return 0.0

    user_prompt = CV_PROMPT.format(
        question=question_text,
        gold_answer=ground_truth,
        llm_response=answer_text,
    )
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # Lower temperature for deterministic judgement
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f" [WARNING] Chat completion request failed: {e}")
        return 0.0

    llm_judge = "UNKNOWN"
    # Parse LLM judge response
    if re.search(r"\b(A|CORRECT)\b", response, re.IGNORECASE):
        acc_reward = 1.0
        llm_judge = "CORRECT"
    elif re.search(r"\b(B|INCORRECT)\b", response, re.IGNORECASE) or re.search(
        r"\b(C|INVALID)\b", response, re.IGNORECASE
    ):
        acc_reward = -1.0
        llm_judge = "INCORRECT"
    else:
        logger.warning(
            f" [WARNING] Judgement format error. \n"
            f"Response: '{response}'\n"
            f"Model Answer: '{answer_text}'\n"
            f"Ground Truth: '{ground_truth}'"
        )
        acc_reward = -1.0

    return acc_reward


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    return compute_score_chart(data_source, solution_str, ground_truth, extra_info)
