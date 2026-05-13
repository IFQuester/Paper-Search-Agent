from typing import Dict

def _search_title(args: Dict):
    """使用浏览器搜索每个年份下各会议的Proceedings，并从中找到符合主题的论文

    Args:
        args (dict): 参数字典，格式：{"topic": "xxx", "start_year": xxxx, "end_year": xxxx, "conferences": ["A", "B", ...]}
    
    Returns:
        list: 搜索结果，格式：[{"id": xx, "title": "xxx", "year": xxxx, "conference": "xxxx", "url": "xxxxx"}, ...]
    """
    pass