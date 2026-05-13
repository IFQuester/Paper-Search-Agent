from typing import List
from langchain_core.messages import BaseMessage

def _extract_args(messages: List[BaseMessage]):
    """抽取论文搜索范围的参数

    Args:
        messages (list of BaseMessage): 历史对话
    
    Returns:
        dict: 参数字典，格式为：{"topic": "xxx", "start_year": xxxx, "end_year": xxxx, "conferences": ["A", "B", ...]}
    """
    pass
    # return {"topic": "Agentic RAG", "start_year": 2023, "end_year": 2026, "conferences": ["ACL", "NAACL", "EMNLP"]}