from typing import Dict

def _download(spath: str, args: Dict):
    """基于找到的论文列表，通过谷歌学术/arXiv下载相应论文

    Args:
        spath (str): 保存目录路径
        args (dict): 要下载的论文列表，格式：[{"id": xx, "title": "xxx", "year": xxxx, "conference": "xxxx", "url": "xxxxx"}, ...]
    
    Returns:
        set: 成功下载，存放成功下载的论文的id
    """
    pass