"""论文下载模块（首版原型）

模块职责：
1. 接收论文结果列表并批量下载 PDF。
2. 对外暴露 `_download(state)` 作为 Agent 下载节点入口，并兼容 `_download(spath, results)`。
3. 首版支持 arXiv、ACL Anthology 与通用开放 PDF 直链。

非目标范围：
1. 目前未接入 Google Scholar。
2. 不做检索逻辑。
"""

from collections.abc import Iterable, Mapping
import os
import re
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse, urlunparse
from urllib.request import Request, urlopen

# ✅️
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.8",
}

# ✅️
_INVALID_FILENAME_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]") # 一共有 9 个非法字符：<>:"/\|?*，以及 ASCII 0-31 的控制字符，这些都是 Windows 文件系统不允许出现在文件名中的。虽然 Linux 和 macOS 更宽松，但为了兼容性，统一替换掉这些字符。
_WHITESPACE_RE = re.compile(r"\s+")

# ─────────────────────────────────────────────────────────────────────────────
# 论文解析器注册表（模块 A）
#
# 背景：上游 state["results"] 中的 url 是平台介绍页（非 PDF 直链），id 是协作者
# 自编的顺序号（非真实论文 id）。因此下载前需要先按论文标题查询平台 API，拿到
# 真实 paper_id 和 PDF 直链。本期仅实现 arxiv 一种解析器；未来扩展只需新增一个
# _resolve_with_xxx 函数并用 @_register_resolver("xxx") 装饰即可。
#
# Resolver 协议：输入 paper dict，成功返回 {"paper_id": str, "pdf_url": str}，
# 失败返回 None（解析器内部自行打印失败原因）。解析器不负责下载、不写文件、
# 不触碰 state，保证高内聚低耦合。
# ─────────────────────────────────────────────────────────────────────────────

Resolver = Callable[[Mapping], Optional[dict]]
_RESOLVERS: dict[str, Resolver] = {}

def _register_resolver(name: str):
    """把一个解析器函数注册到全局 _RESOLVERS 表，便于按名字分发。"""
    def decorator(fn: Resolver) -> Resolver:
        _RESOLVERS[name] = fn
        return fn
    return decorator

@_register_resolver("arxiv")
def _resolve_with_arxiv(paper: Mapping) -> Optional[dict]:
    """通过 arxiv API 按论文标题搜索，返回真实 paper_id 与 pdf_url。

    Args:
        paper: 论文记录字典，至少需要包含可读的 ``title`` 字段。

    Returns:
        成功：``{"paper_id": "2306.05212v1", "pdf_url": "https://arxiv.org/pdf/2306.05212v1"}``
        失败（标题为空 / 库未装 / 网络异常 / 未命中）：``None``，并在控制台打印失败原因。
    """
    title = str(paper.get("title", "")).strip()
    if (not title):
        print("[resolve-arxiv] 标题为空，跳过解析")
        return None

    # 按需 import：即使运行环境暂时缺 arxiv 库，也不影响整个 _download 模块加载
    try:
        import arxiv
    except ImportError as exc:
        print(f"[resolve-arxiv] arxiv 库未安装: {exc}")
        return None

    try:
        client = arxiv.Client()
        # ti:"..." 限定在标题字段做精确短语搜索；max_results=1 只取最相关的一条
        search = arxiv.Search(query=f'ti:"{title}"', max_results=1)
        hits = list(client.results(search))
    except Exception as exc:
        # arxiv 客户端在网络/速率限制/解析异常时会抛各种异常，这里统一兜底
        print(f"[resolve-arxiv] API 查询失败: title={title!r}, error={exc}")
        return None

    if (not hits):
        print(f"[resolve-arxiv] 未找到匹配论文: title={title!r}")
        return None

    top = hits[0]
    return {
        "paper_id": top.get_short_id(),  # 形如 "2306.05212v1"
        "pdf_url": top.pdf_url,           # 形如 "https://arxiv.org/pdf/2306.05212v1"
    }

def _resolve_paper(paper: Mapping) -> Optional[dict]:
    """统一的解析入口；当前固定调度到 arxiv 解析器。

    未来如需多平台路由，可在此处按 ``paper["conference"]`` 等字段选择不同解析器。
    """
    return _RESOLVERS["arxiv"](paper)

# ✅️
def _download(state_or_spath: Any, results: Any = None) -> Any:
    """下载节点入口和后向兼容的原始下载器。

    当调用为 ``_download(state)`` 时，返回 LangGraph state 增量更新，
    这是代理下载节点期望的格式。当调用为 ``_download(save_path, papers)`` 时，
    返回下载成功的论文 id 集合。
    """
    
    # 如果传入了 results 参数，说明是直接调用下载器的场景，state_or_spath 就是 save_path 了
    if (results is not None) or (not isinstance(state_or_spath, Mapping)): 
        return _download_files(state_or_spath, results) # 一般用不到这里
    
    # 否则，兼容 Agent 节点调用，state_or_spath 就是 state 了，这里是正常流程
    state = state_or_spath # 这里的state_or_spath就是字典类型
    papers = state.get("results") or [] #papers是论文结果列表，可能为空，如果没有results这个key，就用空列表代替，保证papers一定是个列表
    injected_papers = None
    ### PAPER_AGENT_DOWNLOAD_DEBUG = 1 # 环境变量控制是否开启调试模式，开启后如果没有搜索结果，就会注入一条模拟论文结果，测试下载流程和总结节点的汇报功能；如果没有开启调试模式，则直接返回空下载结果，测试 summarize 节点在没有下载任何论文时的汇报表现。
    debug_enabled = str(os.getenv("PAPER_AGENT_DOWNLOAD_DEBUG", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if ((not papers) and debug_enabled):
        injected_papers = [
            {
                "id": 90001,
                "title": "Language Models are Few-Shot Learners",
                "year": 2020,
                "conference": "DEBUG",
                "url": "https://arxiv.org/abs/2005.14165",
            }
        ]
        papers = injected_papers
        print(f"[download-node] 调试模式已开启，注入 {len(papers)} 条模拟论文结果")

    if (not papers):
        return {"downloaded": set()} # 空下载结果，供 summarize 节点测试在没有下载任何论文时的汇报表现

    print(f"[download-node] 正在下载，候选论文数: {len(papers)}")

    try:
        result = _download_files(state["save_path"], papers)
    except Exception as exc:
        print(f"[download-node] 下载异常: {exc}")
        return {"downloaded": set()}

    if (isinstance(result, set)):
        if (injected_papers is not None):
            return {"results": injected_papers, "downloaded": result}
        return {"downloaded": result}

    if (result is None):
        if (injected_papers is not None):
            return {"results": injected_papers, "downloaded": set()}
        return {"downloaded": set()}

    try:
        normalized_result = set(result)
    except TypeError:
        normalized_result = set()

    if (injected_papers is not None):
        return {"results": injected_papers, "downloaded": normalized_result}
    return {"downloaded": normalized_result}

# ✅️
def _download_files(spath: str, results: Any) -> set[Any]:
    """批量下载论文 PDF，并返回成功下载的论文 id 集合。

    流程：
      1. 规整输入论文列表。
      2. 对每条论文先调用 ``_resolve_paper`` 拿到真实 PDF 直链。
      3. 用现有候选生成 + PDF 下载校验逻辑完成下载。
      4. 解析失败 / 候选 URL 不可用 / 下载失败的论文，仅在控制台打印失败信息
         （含批次末尾的失败 id 汇总），不进入返回的 set，也不写入 state。

    Args:
        spath (str): 保存目录路径。
        results (Any): 论文列表，格式：
            [{"id": xx, "title": "xxx", "year": xxxx, "conference": "xxxx", "url": "xxxxx"}, ...]

    Returns:
        set[Any]: 成功下载的论文 id 集合。
    """
    downloaded_ids: set[Any] = set() # 初始为空集合，存储成功下载的论文 id
    failed_ids: list = []            # 仅用于批次末尾汇总打印，不返回
    papers = _normalize_results(results)
    if (not papers):
        return downloaded_ids

    try:
        os.makedirs(spath, exist_ok=True)
    except OSError as exc:
        print(f"[download] 创建目录失败: {spath}, error={exc}")
        return downloaded_ids

    for index, paper in enumerate(papers):
        paper_info = _extract_paper_info(paper, index)
        if (paper_info is None):
            continue

        paper_id, title, _platform_url = paper_info # 原 url 是平台介绍页，不再用于构造下载链

        # 模块 A：解析真实 PDF 直链（当前走 arxiv）
        resolved = _resolve_paper(paper)
        if (resolved is None):
            print(f"[download] 解析失败: id={paper_id}, title={title!r}")
            failed_ids.append(paper_id)
            continue

        pdf_url = resolved["pdf_url"]
        # 复用现有候选生成逻辑：直链 PDF 会原样作为单元素列表返回，未来若解析器
        # 返回的链接需要变体补丁（如 .pdf 后缀补齐），也能继续在这里收口处理。
        candidate_urls = _build_candidate_urls(pdf_url)
        if (not candidate_urls):
            print(f"[download] 解析后 URL 不可用: id={paper_id}, url={pdf_url}")
            failed_ids.append(paper_id)
            continue

        file_stem = _build_file_stem(paper_id, title)
        success = False
        for candidate_url in candidate_urls:
            pdf_bytes = _fetch_pdf_bytes(candidate_url)
            if (pdf_bytes is None):
                continue

            target_path = _next_available_pdf_path(spath, file_stem)
            try:
                with open(target_path, "wb") as fw:
                    fw.write(pdf_bytes)
            except OSError as exc:
                print(f"[download] 写文件失败: {target_path}, error={exc}")
                break

            downloaded_ids.add(paper_id)
            print(f"[download] 下载成功: id={paper_id}, file={target_path}")
            success = True
            break

        if (not success):
            print(f"[download] 下载失败: id={paper_id}, title={title!r}")
            failed_ids.append(paper_id)

    if (failed_ids):
        print(f"[download] 本批次失败 id 汇总（{len(failed_ids)} 条）: {failed_ids}")
    return downloaded_ids

# ✅️
def _normalize_results(results: Any) -> list[Any]:
    """将输入统一转换为可迭代论文列表。"""
    if (results is None):
        return []

    if (isinstance(results, Mapping)):
        # 兼容误传单条记录的场景。
        if ("id" in results) and ("url" in results):
            return [results] # 先最小标准，后面会有对应的处理
        return [] # 如果是字典但不符合论文记录格式，则返回空列表

    if (isinstance(results, (str, bytes))):
        return [] # _download.py 里的 _extract_paper_info，它要求每一项至少得像个字典，得能拿到 id、url、title。可如果前面传进来的是字符串，被拆开后每一项就是一个字符，比如 h，那当然不是字典，于是全都会被跳过。

    if (not isinstance(results, Iterable)):
        return []

    return list(results)

# ✅️
def _extract_paper_info(paper: Any, index: int) -> tuple[Any, str, str] | None:
    """就是填一下属性，做一点健壮性检查"""
    if (not isinstance(paper, Mapping)):
        return None

    # 提取而已
    paper_id = paper.get("id")
    paper_url = str(paper.get("url", "")).strip()
    title = str(paper.get("title", "")).strip()

    if (paper_id is None) or (not paper_url):
        return None

    if (not title):
        title = f"paper_{index + 1}"

    return paper_id, title, paper_url

# ✅️
def _build_candidate_urls(raw_url: str) -> list[str]:
    """根据来源生成候选下载链接。

    首版策略：
    1. 原始链接始终作为候选。
    2. arXiv 补充标准 PDF 链接。
    3. ACL Anthology 页面链接补充 `.pdf` 直链。
    """
    parsed = urlparse(raw_url)
    if (parsed.scheme not in {"http", "https"}):
        return [] # 只处理 http/https 链接，其他协议不予下载

    candidates: list[str] = []
    _append_unique(candidates, raw_url)

    host = parsed.netloc.lower()
    path = parsed.path or ""

    if ("arxiv.org" in host):
        arxiv_id = _extract_arxiv_id(path) # （这里是健壮性....不过，也可能有点多此一举..）从路径中提取 arXiv 论文 id，如果成功提取到 id，就构造标准 PDF 链接并添加到候选列表中
        if (arxiv_id):
            _append_unique(candidates, f"https://arxiv.org/pdf/{arxiv_id}.pdf")

    if ("aclanthology.org" in host):
        normalized_path = path.rstrip("/") # 去掉路径末尾的斜杠，防止出现 /P19-1001/ 这种情况
        if normalized_path and (not normalized_path.lower().endswith(".pdf")):
            pdf_url = urlunparse(
                ParseResult(
                    scheme=parsed.scheme,
                    netloc=parsed.netloc,
                    path=f"{normalized_path}.pdf",
                    params="",
                    query="",
                    fragment="",
                )
            )
            _append_unique(candidates, pdf_url)

    return candidates

# ✅️
def _extract_arxiv_id(path: str) -> str | None:
    """从 arXiv 路径中提取论文 id。"""
    normalized = path.strip() 
    if (not normalized):
        return None
    # 为什么 arXiv 老是用这个？
    # 因为这是 arXiv 自己很多年沿用下来的网址规则。它把不同页面分开了：
    # /abs/论文编号--看论文介绍页，也就是摘要页
    # /pdf/论文编号.pdf--直接拿 PDF 文件
    if normalized.startswith("/abs/"):
        arxiv_id = normalized[len("/abs/") :] # 字符串切片，去掉开头的 /abs/，剩下的就是论文编号了
    elif normalized.startswith("/pdf/"):
        arxiv_id = normalized[len("/pdf/") :] # 字符串切片，去掉开头的 /pdf/，剩下的就是论文编号了
    else:
        return None

    arxiv_id = arxiv_id.strip("/") # 去掉两端的斜杠，防止出现 /2101.00001/ 这种情况
    if arxiv_id.lower().endswith(".pdf"): # 如果以 .pdf 结尾，去掉 .pdf 后缀
        arxiv_id = arxiv_id[:-4]

    arxiv_id = arxiv_id.strip() # 最后再去掉两端的空白，防止出现 " 2101.00001 " 这种情况
    if (not arxiv_id):
        return None

    return arxiv_id

# ✅️
def _append_unique(items: list[str], value: str) -> None:
    """按顺序去重追加。"""
    if value and (value not in items):
        items.append(value)

# ✅️
def _fetch_pdf_bytes(url: str, timeout: float = 25) -> bytes | None:
    """从URL下载内容并验证响应体是有效的PDF。

    执行GET请求（带有浏览器模拟头部），然后使用魔数（``%PDF-``）和
    ``Content-Type``头部来验证payload，以最小化假正例。

    Args:
        url: 要下载的目标URL。
        timeout: 请求超时时间（秒）。接受小数值（例如``10.5``）。
                 默认为25。

    Returns:
        当响应被识别为PDF时返回原始PDF字节，或在任何网络/协议错误
        或服务器返回非PDF内容时返回``None``。
    """
    try:
        request = Request(url=url, headers=_REQUEST_HEADERS, method="GET")
        with urlopen(request, timeout=timeout) as response:
            payload: bytes = response.read() # read的作用 
            content_type: str = str(
                response.headers.get("Content-Type", "")
            ).lower()
    except HTTPError as exc:
        print(f"[download] HTTP {exc.code} {exc.reason}: {url}")
        return None
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"[download] 请求失败: {url}, error={exc}")
        return None

    if _is_pdf_payload(payload, content_type):
        return payload

    print(f"[download] 非 PDF 响应, 跳过: {url}")
    return None

# ✅️
def _is_pdf_payload(payload: bytes, content_type: str) -> bool:
    """验证 *payload* 是否是（或自称是）PDF。

    使用两步检查来减少误判：
    1. 魔数检查——内容头部必须以 ``%PDF-`` 开头。
    2. Content-Type 头检查——必须是 ``application/pdf``，
       或以 ``application/pdf;`` 开头。
    """
    if not payload:
        return False

    if payload.startswith(b"%PDF-"):
        return True

    return content_type == "application/pdf" or content_type.startswith("application/pdf;")

# ✅️
def _build_file_stem(paper_id: Any, title: str) -> str:
    """构造稳定、可读的文件名主干。"""
    id_part = _sanitize_filename(str(paper_id))
    title_part = _sanitize_filename(title)

    if id_part and title_part:
        stem = f"{id_part}_{title_part}" # 例如 2101.00001_An_Interesting_Paper
    else:
        stem = id_part or title_part or "paper" # 例如 2101.00001 或 An_Interesting_Paper，甚至 paper（如果 id 和 title 都没有的话）

    stem = stem.strip("._ ") # 例如 2101.00001_An_Interesting_Paper，去掉两端的点、下划线和空格，防止出现 .2101.00001_An_Interesting_Paper. 这种情况
    if (not stem):
        stem = "paper"

    # 控制文件名长度，避免路径过长问题。
    return stem[:150].rstrip("._ ")

# ✅️
def _sanitize_filename(value: str) -> str:
    """清洗文件名中的非法字符与多余空白。"""
    cleaned = _INVALID_FILENAME_RE.sub("_", value)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip() # 替换连续空白为单个空格，并去掉两端空白
    cleaned = cleaned.strip(".") # 去掉两端的点，避免 Windows 上出现 .pdf. 这种情况，导致文件扩展名识别错误
    return cleaned

# ✅️
def _next_available_pdf_path(directory: str, file_stem: str) -> str:
    """生成不覆盖已有文件的保存路径。"""
    base_path = os.path.join(directory, f"{file_stem}.pdf")
    if (not os.path.exists(base_path)):
        return base_path

    index = 1
    while True:
        candidate = os.path.join(directory, f"{file_stem}_{index}.pdf")
        if (not os.path.exists(candidate)):
            return candidate
        index += 1


# ✅️
def download_with_debug(save_path: str, papers: Any, debug_enabled: bool = False) -> tuple[set, Any]:
    """Agent Node 专用的下载接口，带调试模式支持。

    基于论文结果列表下载开放论文 PDF，支持注入模拟论文用于测试。
    返回 (downloaded_ids, injected_papers) 元组，供 summarize 节点使用。

    Args:
        save_path (str): 保存目录路径
        papers (Any): 论文列表，可为空
        debug_enabled (bool): 是否开启调试模式。开启时若无搜索结果，则注入模拟论文

    Returns:
        tuple[set, Any]: (成功下载的论文 id 集合, 注入的模拟论文列表或 None)
    """
    papers = papers or []
    injected_papers = None

    # 如果没有搜索结果但调试模式开启，则注入一条模拟论文结果，测试下载流程和总结节点的汇报功能
    if ((not papers) and debug_enabled):
        injected_papers = [
            {
                "id": 90001,
                "title": "Language Models are Few-Shot Learners",
                "year": 2020,
                "conference": "DEBUG",
                "url": "https://arxiv.org/abs/2005.14165",
            }
        ]
        papers = injected_papers
        print(f"[download-node] 调试模式已开启，注入 {len(papers)} 条模拟论文结果")

    # 直接返回空下载结果，测试 summarize 节点在没有下载任何论文时的汇报表现
    if (not papers):
        return set(), None

    print(f"[download-node] 正在下载，候选论文数: {len(papers)}")

    try:
        result = _download_files(save_path, papers)
    except Exception as exc:
        print(f"[download-node] 下载异常: {exc}")
        return set(), injected_papers

    if (isinstance(result, set)):
        return result, injected_papers

    if (result is None):
        return set(), injected_papers

    try:
        normalized_result = set(result)
    except TypeError:
        normalized_result = set()

    return normalized_result, injected_papers


# # ─────────────────────────────────────────────────────────────────────────────
# # 本地手动测试入口
# #
# # 用途：直接 `python utils/_download.py` 跑全量测试 JSON，验证"解析+下载"链路
# # 是否正常。投入正式使用时可注释掉本块，不影响 _download 被 agent.py 调用。
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import json
#     import sys

#     # Windows 上 Python 默认 stdout 走 GBK/cp936，从 Git Bash 等 UTF-8 终端
#     # 读取会显示成乱码。仅在直跑测试入口时强制改成 UTF-8，不影响被 agent.py 调用的场景。
#     # 用 getattr 取属性是因为静态类型存根里 sys.stdout 标的是 TextIO（没有 reconfigure），
#     # 实际运行时是 TextIOWrapper（Python 3.7+ 自带 reconfigure）。
#     for stream in (sys.stdout, sys.stderr):
#         reconfigure = getattr(stream, "reconfigure", None)
#         if callable(reconfigure):
#             try:
#                 reconfigure(encoding="utf-8")
#             except OSError:
#                 pass

#     here = os.path.dirname(os.path.abspath(__file__))
#     test_json = os.path.normpath(os.path.join(here, "..", "测试的输入数据.json"))
#     save_dir = os.path.normpath(os.path.join(here, "..", "save"))

#     with open(test_json, "r", encoding="utf-8") as fr:
#         all_papers = json.load(fr)

#     state = {"results": all_papers, "save_path": save_dir}
#     result = _download(state)

#     downloaded = result.get("downloaded", set())
#     print("=" * 60)
#     print(f"输入数量: {len(all_papers)}")
#     print(f"成功下载 {len(downloaded)} 条: {sorted(downloaded)}")
#     print(f"PDF 保存到: {save_dir}")