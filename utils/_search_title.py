import os
import json
import dotenv
import requests
from typing import Dict, List, Any, Optional
from collections import Counter
from bs4 import BeautifulSoup
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()

SERPER_URL = "https://google.serper.dev/search"
MODEL_PATH = "models/bge-small-zh-v1.5/"
BATCH_SIZE = 1024
SPLIT = 10

class SelectedSite(BaseModel):
    num: int = Field(description="最符合要求的网页序号", default=None)
    reason: Optional[str] = Field(description="选择该网页的理由")

class SelectedTag(BaseModel):
    judge: bool = Field(description="True表示可以借助提取该标签找到所有论文标题，否则为False", default=False)
    reason: Optional[str] = Field(description="作出判断的理由")
    num: Optional[int] = Field(description="某一个包含了论文标题的片段的序号")
    # code: Optional[str] = Field(description="一段可使用eval()执行的Python代码片段，用于从bs4.element.Tag对象列表`tags`中提取出论文标题列表，放在名为`titles`的字符串列表中")

class Parsing(BaseModel):
    think: Optional[str] = Field(description="对如何从HTML片段中解析出论文标题的分析")
    code: Optional[str] = Field(description="用于解析论文标题的Python代码片段")

class Filter(BaseModel):
    think: Optional[str] = Field(description="对哪些论文符合所给主题的分析")
    numbers: List[int] = Field(description="符合所给主题的论文序号", default=[])


def _filter_urls(year: int, conference: str, urls: List[Dict[str, Any]]):
    """使用LLM筛选Google上搜索到的论文录用网站
    Args:
        year (int): 会议年份
        conference (str): 会议名称
        urls (list): 网页列表，格式为：[{"title": "xxx", "link": "xxx", "snippet": "xxx", "position": xx}]
    
    Returns:
        dict: 筛选结果，格式与urls的元素相同
    """
    # print(f"year: {year}\n conference: {conference}\n urls: {urls}\n")
    llm = ChatDeepSeek(
        model = "deepseek-chat",
        max_tokens=2048,
        api_key=os.getenv("DEEPSEEK_KEY")
    )        
    llm = llm.with_structured_output(SelectedSite)
    formatted = [("【序号】："+str(i)+"\n")+("标题："+site['title']+"\n")+("内容片段："+site["snippet"]+"\n") for i, site in enumerate(urls)]
    formatted = "\n".join(formatted)
    prompt = HumanMessage(
        content=(
            "给你一组网页的标题和内容片段，请你筛选出最符合要求的那一条。"
            f"要求：{conference}会议在{year}年的录用论文（Accepted Papers）网页，这个网页应该包含了当年该会议所录用论文的标题，且应当是官方而非第三方的网站。"
            f"候选网页：\n{formatted}\n"
            "最后请按输出最符合要求的网页的序号及选择它的理由。"
            )
    )
    # print(prompt.content)
    result = llm.invoke([prompt])
    print(result)

    return urls[result.num]


def _extract_titles(doc: str):
    """使用 启发式规则+LLM 从录用论文网页的HTML文件中解析出所有录用论文的标题

    Args:
        doc (str): 录用论文网页的HTML文档
    
    Returns:
        list: 字符串列表，即抽取结果
    """
    llm = ChatDeepSeek(
        model = "deepseek-chat",
        max_tokens=2048,
        api_key=os.getenv("DEEPSEEK_KEY")
    )
    llm_select = llm.with_structured_output(SelectedTag)
    llm_parsing = llm.with_structured_output(Parsing)

    soup = BeautifulSoup(doc)
    all_tags = soup.find_all(True)
    tag_counts = Counter(tag.name for tag in all_tags)
    top_5 = tag_counts.most_common(5)
    # print(top_5)
    for name, cnt in top_5:
        tags = soup.find_all(name)
        sampled = []
        for i in range(max(0, cnt//2-2), min(cnt//2+3, cnt), 1):
            sampled.append(tags[i])
        formatted = [f"【片段{i+1}】\n"+str(tag) for i, tag in enumerate(sampled)]
        formatted = "\n\n".join(formatted)
        # print(formatted)
        select_prompt = HumanMessage(
            content=(
                "我们获取了某会议的录用论文（Accepted Papers）网页的HTML文档，"
                f"解析出所有<{name}>标签包裹起来的HTML片段，并从采样出一部分。"
                "我们假设该网页中每一篇录用论文的信息都用相同的HTML标签来排版。"
                f"请你基于这些片段，判断是否可以通过捕获所有<{name}>标签，来找到所有录用论文的标题。"
                f"我们能接受解析出所有<{name}>标签后，会找到一些无关的内容（其中不包含录用论文的标题），"
                f"但解析出所有<{name}>标签，一定要找到所有录用论文的标题。"
                f"由<{name}>标签包裹起来的HTML片段：\n\n{formatted}\n\n"
                "输出判断结果（True/False）及理由。如果判断为真，再输出任意一个包含了论文标题的片段的序号。"
            )
        )
        result = llm_select.invoke([select_prompt])
        # print(result)
        
        if result.judge:
            parse_prompt = HumanMessage(
                content=(
                    "现在有一个Tag（bs4.element.Tag）对象列表，名为`tags`，"
                    "其中大多数Tag对象的标签排版类似，且均包含一篇论文的标题。"
                    "请你写一个可以使用eval()函数执行的Python代码片段，"
                    "从每个Tag对象中解析出论文标题（不包括作者），并存放到一个名为`titles`的字符串列表中。\n"
                    f"Tag对象的示例如下：\n\n{str(sampled[result.num-1])}\n\n"
                    "由于并非`tags`列表中的每一个Tag对象都与示例相同，你编写的代码应具有鲁棒性，能直接略过那些排版不相符的Tag对象而不报错。"
                )
            )
            # print(parse_prompt)
            parse_result = llm_parsing.invoke([parse_prompt])
            # print(parse_result)
            my_globals, my_locals = {"tags": tags}, {}
            exec(parse_result.code, my_globals, my_locals)
            titles = my_locals["titles"] if "titles" in my_locals else []
            break
    
    return titles


def _filter_papers(papers: List[str], topic: str):
    """使用 Embedding+LLM 实现 Recall+Rank 双阶段检索策略

    Args:
        papers (list): 论文标题列表
        topic (str): 研究主题
    
    Returns:
        list: 过滤后的论文标题列表
    """
    embedding = HuggingFaceEmbeddings(
        model_name=str(MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name="new_collection",
        embedding_function=embedding,
    )
    docs = [Document(page_content=paper) for paper in papers]
    for i in range(0, len(docs), BATCH_SIZE):
        vector_store.add_documents(docs[i:min(i+BATCH_SIZE, len(docs))])

    llm = ChatDeepSeek(
        model = "deepseek-chat",
        max_tokens=2048,
        api_key=os.getenv("DEEPSEEK_KEY")
    )
    llm = llm.with_structured_output(Filter)
    
    results = []
    retrieved = vector_store.similarity_search(query=topic, k=50)
    for i in range(0, len(retrieved), SPLIT):
        split = [doc.page_content for doc in retrieved[i:min(i+SPLIT, len(retrieved))]]
        formatted = [f"【序号】：{id+1}\n"+f"标题：{title}" for id, title in enumerate(split)]
        formatted = "\n\n".join(formatted)
        prompt = HumanMessage(
            content=(
                "给你一组论文的标题和一个研究主题，找出符合研究主题的论文。\n"
                f"研究主题：{topic}\n"
                f"候选论文：\n\n{formatted}\n\n"
                "请输出分析过程和分析后你认为符合所给主题的论文的序号。若都不符合主题，则输出空列表。"
            )
        )
        result = llm.invoke([prompt])
        # print(result)
        results.extend([split[id-1] for id in result.numbers])
    print(results)
    return results



def _search_title(args: Dict):
    """使用浏览器搜索每个年份下各会议的Proceedings，并从中找到符合主题的论文

    Args:
        args (dict): 参数字典，格式：{"topic": "xxx", "beg_year": xxxx, "end_year": xxxx, "conferences": ["A", "B", ...]}
    
    Returns:
        list: 搜索结果，格式：[{"id": xx, "title": "xxx", "year": xxxx, "conference": "xxxx", "url": "xxxxx"}, ...]
    """
    topic = args["topic"]
    beg_year, end_year = args["beg_year"], args['end_year']
    conferences = args['conferences'] 
    cnt, results = 0, []
    for year in range(beg_year, end_year+1, 1):
        for conference in conferences:
            # 在Google上搜索当年该会议的Accepted Papers网页
            query = f"{conference} {year} Accepted Papers"
            payload = {
                "q": query,
                "page": 1
            }
            headers = {
                'X-API-KEY': os.environ["SERPER_KEY"],
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", SERPER_URL, headers=headers, json=payload)
            result = response.json()
            # print(result['organic'])
            
            # 过滤出最符合要求的那一则网页
            valid_url = _filter_urls(year, conference, result["organic"])
            print(valid_url)
            
            # 从网页中抽取出所有论文的标题
            paper_titles = []
            if valid_url:
                response = requests.get(valid_url["link"])
                doc = response.text
                print(f"Document length: {len(doc)}")

                paper_titles.extend(_extract_titles(doc))
            print(f"抽取出 {len(paper_titles)} 篇论文，前5条：{paper_titles[:5]}")

            # 从论文中筛选出符合主题 (Topic) 的那部分
            filtered_papers = _filter_papers(paper_titles, topic)
            for paper in filtered_papers:
                cnt += 1
                results.append({"id": cnt, "title": paper, "year": year, "conference": conference, "url": valid_url["link"]})
            print(f"最终筛选出 {len(filtered_papers)} 篇符合主题的论文，前5条：{filtered_papers[:5]}")

    return result
            

if __name__ == "__main__":
    args = {"topic": "Retrieval-Augmented Generation", "beg_year": 2025, "end_year": 2025, "conferences": ["ACL"]}
    # args = {"topic": "Retrieval-Augmented Generation", "beg_year": 2024, "end_year": 2024, "conferences": ["NAACL"]}
    # args = {"topic": "Retrieval-Augmented Generation", "beg_year": 2023, "end_year": 2023, "conferences": ["NAACL"]}
    # args = {"topic": "Weather Classification", "beg_year": 2023, "end_year": 2023, "conferences": ["CVPR"]}
    _search_title(args)
    # requests.post()
    # https://2025.aclweb.org/program/main_papers/
    # https://sigir2025.dei.unipd.it/accepted-papers.html
    # response = requests.get("https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers")
    # response = requests.get("https://2025.aclweb.org/program/main_papers/")
    # response = requests.get("https://sigir2025.dei.unipd.it/accepted-papers.html")
    # with open("save/cvpr2025.html", mode='r') as f:
        # doc = f.read()
    # print("Get response!")
    # soup = BeautifulSoup(doc)
    # print(type(soup))
    # all_tags = soup.find_all(True)
    # x = (tag.name for tag in all_tags)
    # print(type(x))
    # tag_counts = Counter(tag.name for tag in all_tags)
    # print(tag_counts)
    # top_3 = tag_counts.most_common(3)
    # print(top_3)
    # for name, cnt in top_3:
    #     tags = soup.find_all(name)
    #     print(name, "\n", tags[cnt//2])
    # tags = soup.find_all("strong")
    # print(tags[0].string)
    # print(response.text)
    # with open("save/sigir2025.html", mode="w") as f:
        # f.write(response.text)
