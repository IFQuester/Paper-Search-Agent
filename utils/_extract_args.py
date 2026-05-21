"""论文检索参数提取模块。

模块职责：
1. 从历史对话中提取论文检索参数。
2. 先尝试 LLM 结构化抽取，失败后回退到规则抽取。
3. 对输出进行统一归一化，保证调用方拿到稳定格式。

输入契约：
- messages: Sequence[BaseMessage]，历史对话消息列表。

输出契约（固定四键，不可破坏）：
- topic: str
- start_year: int
- end_year: int
- conferences: List[str]

默认策略：
- 年份缺失时：start_year=end_year=当前年。
- 会议缺失时：使用 DEFAULT_CONFERENCES。

非目标范围：
- 不做会议合法性校验。
- 不依赖下载模块或标题检索模块。
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr

# ✅️
YEAR_LOWER_BOUND = 1900
YEAR_UPPER_FUTURE_OFFSET = 1 # 允许用户指定未来一年的论文，考虑到有些会议会提前发布下一年的 CFP 和论文列表，这样用户就可以直接搜索下一年的论文，而不必等到明年才来搜索；同时，在年份归一化时，我们也会允许 end_year 最多比当前年大一年，以适应这种情况；如果用户指定的年份超过了这个范围，我们会进行合理的调整，确保最终返回的年份在合理范围内。

# ✅️
DEFAULT_CONFERENCES = [
    "ACL",
    "EMNLP",
    "NAACL",
    "COLING",
    "AAAI",
    "IJCAI",
    "ICML",
    "NeurIPS",
    "ICLR",
    "KDD",
    "WWW",
]

# ✅️
_CONFERENCE_ALIASES = {
    "acl": "ACL",
    "association for computational linguistics": "ACL",
    "emnlp": "EMNLP",
    "conference on empirical methods in natural language processing": "EMNLP",
    "naacl": "NAACL",
    "north american chapter of the association for computational linguistics": "NAACL",
    "coling": "COLING",
    "international conference on computational linguistics": "COLING",
    "aaai": "AAAI",
    "aaai conference on artificial intelligence": "AAAI",
    "ijcai": "IJCAI",
    "international joint conference on artificial intelligence": "IJCAI",
    "icml": "ICML",
    "international conference on machine learning": "ICML",
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "neural information processing systems": "NeurIPS",
    "conference on neural information processing systems": "NeurIPS",
    "iclr": "ICLR",
    "international conference on learning representations": "ICLR",
    "kdd": "KDD",
    "knowledge discovery and data mining": "KDD",
    "www": "WWW",
    "world wide web conference": "WWW",
    "the web conference": "WWW",
    "thewebconf": "WWW",
    "webconf": "WWW",
}

# ✅️
_CONFERENCE_ALIASES_NORMALIZED = {
    re.sub(r"[^a-z0-9]", "", alias.lower()): canonical
    for alias, canonical in _CONFERENCE_ALIASES.items()
}

# ✅️
_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

# ✅️
class ExtractArgsResult(TypedDict):
    """参数提取结果契约。

    保持与调用方约定一致：必须返回四个键，且类型稳定。
    """

    topic: str
    start_year: int
    end_year: int
    conferences: List[str]

# ✅️
class _ExtractedArgs(BaseModel):
    """LLM 结构化抽取的中间模型。"""

    topic: str = Field(default="")
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    conferences: List[str] = Field(default_factory=list)

# ✅️
def _extract_args(messages: Sequence[BaseMessage]) -> ExtractArgsResult:
    """抽取论文搜索范围的参数

    Args:
        messages (list of BaseMessage): 历史对话
    
    Returns:
        dict: 参数字典，格式为：{"topic": "xxx", "start_year": xxxx, "end_year": xxxx, "conferences": ["A", "B", ...]}
    """
    # 一次调用只负责“抽取 + 归一化”，不做外部流程控制。
    # 对 Agent 链路兜底：即使某个抽取环节异常，也返回稳定四键结构。
    safe_messages = list(messages or []) # 防止 None 输入，且后续可能需要多次遍历消息列表，先转换为列表。
    history = _collect_history(safe_messages) # 转换为 (role, text) 结构，方便后续处理。
    merged_text = _merge_human_messages(history) # 加入\n分隔符合并人类消息

    llm_result: Dict[str, Any] = {}
    try:
        candidate = _extract_with_llm(history)
        if isinstance(candidate, dict):
            llm_result = candidate
    except Exception:
        llm_result = {} # LLM 调用失败或返回不规范结果时，保持 llm_result 为空字典，让流程继续走规则回退。

    rule_result: Dict[str, Any] = {}
    if _needs_rule_fallback(llm_result): # 判断是否需要规则回退：当 LLM 结果缺失或不完整时，才启用规则抽取；如果 LLM 已经成功抽取了部分参数，我们就先看看这些参数是否足够用，如果不够用再启用规则抽取来补全。
        try:
            candidate = _extract_with_rules(merged_text)
            if isinstance(candidate, dict):
                rule_result = candidate
        except Exception:
            rule_result = {}

    try:
        return _normalize_args(llm_result, rule_result, merged_text)
    except Exception:
        return _default_result(merged_text)

# ✅️
def _needs_rule_fallback(llm_result: Dict[str, Any]) -> bool:
    """判断是否需要规则回退。"""

    if not llm_result:
        return True

    topic = _clean_topic(str(llm_result.get("topic", "")))
    start_year = _to_int(llm_result.get("start_year"))
    end_year = _to_int(llm_result.get("end_year"))
    conferences = _normalize_conferences(llm_result.get("conferences", []))

    return not (topic and start_year is not None and end_year is not None and conferences)

# ✅️
def _default_result(merged_text: str) -> ExtractArgsResult:
    """极端异常下的最终保底结果。"""

    now_year = datetime.now().year
    topic = _clean_topic(merged_text) or "未指定主题"
    return {
        "topic": topic,
        "start_year": now_year,
        "end_year": now_year,
        "conferences": DEFAULT_CONFERENCES.copy(),
    }

# ✅️
def _collect_history(messages: Sequence[BaseMessage]) -> List[Tuple[str, str]]:
    """将消息列表转换为 (role, text) 的轻量历史。"""
   
    history: List[Tuple[str, str]] = [] # 用元组是因为角色和文本是天然的两列信息，使用元组能更清晰地表达这个结构，同时也避免了不必要的字典层级和键名占用。
    for message in messages:
        content = _safe_message_content(getattr(message, "content", ""))
        if not content:
            continue
        role = getattr(message, "type", "unknown")
        history.append((role, content))
    return history

# ✅️
def _safe_message_content(content: Any) -> str:
    """统一处理消息 content 的多种格式。"""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                chunks.append(str(item["text"]))
        return " ".join(chunks).strip()

    return str(content).strip()

# ✅️
def _merge_human_messages(history: Sequence[Tuple[str, str]]) -> str:
    """优先合并人类消息；若无则回退到全量文本。"""
    
    
    human_texts = [text for role, text in history if role == "human"] # 拿出所有人类消息的文本，合并成一个字符串，作为后续抽取的输入。优先使用人类消息是因为它们更可能包含明确的参数描述，而系统消息可能更多是上下文或辅助信息。
    if human_texts:
        return "\n".join(human_texts) # 将所有人类消息用换行符连接起来，形成一个整体文本，供后续抽取使用。

    all_texts = [text for _, text in history]
    return "\n".join(all_texts)

# ✅️
def _extract_with_llm(history: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """使用 LLM 做结构化参数抽取。

    若缺少密钥、无历史消息或调用失败，返回空字典并交由规则回退。
    """

    api_key = os.getenv("DEEPSEEK_KEY", "").strip()
    if not api_key or not history:
        return {}

    # 延迟导入，避免模块导入阶段拉起重依赖链，保证规则回退路径可测。
    try:
        from langchain_deepseek import ChatDeepSeek
    except Exception:
        return {}

    dialogue = "\n".join([f"{role}: {text}" for role, text in history[-12:]]) # ⚠️只保留最近12条消息（12 条 ≈ 6轮来回），避免输入过长导致调用失败；同时保留角色信息，帮助 LLM 理解对话结构。
    llm = ChatDeepSeek(
        model="deepseek-chat",
        max_tokens=2048, # ⚠️增加最大 token 数量，避免长对话被截断
        temperature=0,
        api_key=SecretStr(api_key),
    )
    structured_llm = llm.with_structured_output(_ExtractedArgs)

    system_prompt = SystemMessage(
        content=(
            "你是参数抽取器。请从中英文对话历史中抽取论文检索参数。"
            "字段固定为 topic、start_year、end_year、conferences。"
            "字段缺失时可留空，不要臆造。conferences 使用会议简称。"
        )
    )
    user_prompt = HumanMessage(content=f"对话历史如下：\n{dialogue}")

    # 调用可能失败，或返回不规范结果；都不应抛异常，而是返回空字典，让流程继续走规则回退。
    try:
        result = structured_llm.invoke([system_prompt, user_prompt])
        if isinstance(result, _ExtractedArgs):
            return result.model_dump() # 将 Pydantic 模型转换为字典，供后续处理；如果 LLM 成功抽取并返回了结构化结果，我们就把它转换成一个普通的字典格式，这样后续的归一化函数就可以统一处理这个结果，无论它是来自 LLM 还是规则抽取。
        if isinstance(result, dict):
            return result
        return {}
    except Exception:
        return {}

# ✅️
def _extract_with_rules(text: str) -> Dict[str, Any]:
    """规则抽取兜底：从文本中解析 topic、year、conference。"""

    now_year = datetime.now().year # 获取当前年份，供相对年份解析使用。
    start_year, end_year = _extract_year_range(text, now_year)
    conferences = _extract_conferences(text)
    topic = _extract_topic(text)
    return {
        "topic": topic,
        "start_year": start_year,
        "end_year": end_year,
        "conferences": conferences,
    }

# ✅️
def _extract_year_range(text: str, now_year: int) -> Tuple[Optional[int], Optional[int]]:
    """抽取年份范围，兼容中英文表达。"""

    cleaned = (
        text.replace("～", "~")
        .replace("—", "-")
        .replace("–", "-")
        .replace("至", "到")
    )
    lowered = cleaned.lower()

    range_patterns = [
        r"((?:19|20)\d{2})\s*年?\s*[-~到]\s*((?:19|20)\d{2})\s*年?",
        r"(?:from|between)\s+((?:19|20)\d{2})\s+(?:to|and)\s+((?:19|20)\d{2})",
    ]
    for pattern in range_patterns:
        matched = re.search(pattern, cleaned) # 这里的正则模式需要兼容多种分隔符和表达方式，确保能捕获到用户可能输入的各种年份范围格式；同时要注意年份前后的可选“年”字，以及英文表达中的关键词。
        if matched: # 这里匹配的是一个年份范围，如果匹配成功，我们就可以从匹配结果中提取出两个年份，作为初步的年份范围；如果用户输入了不合理的年份范围（比如结束年份早于开始年份），我们还需要进行调整，确保最终返回的年份范围是合理的。
            # group(1) 和 group(2) 分别对应正则中第一个和第二个捕获组，
            # 也就是两个年份；我们先尝试从匹配结果中提取这两个年份，并转换为整数；
            # 如果转换成功，我们就得到了一个初步的年份范围。
            y1 = int(matched.group(1)) 
            y2 = int(matched.group(2))
            if y1 > y2:
                y1, y2 = y2, y1
            return y1, y2
    # 中文
    relative = re.search(r"(?:最近|近)\s*([一二两三四五六七八九十\d]+)\s*年", cleaned)
    if relative:
        years = _to_int(relative.group(1)) # (?:最近|近) 用了 ?:，是非捕获组，不占编号
        if years and years > 0:
            return now_year - years + 1, now_year
    # 英文
    relative_en = re.search(r"(?:last|past)\s*(\d{1,2})\s*years?", lowered)
    if relative_en:
        years = _to_int(relative_en.group(1)) # (?:last|past) 用了 ?:，是非捕获组，不占编号
        if years and years > 0:
            return now_year - years + 1, now_year

    if "今年" in cleaned:
        return now_year, now_year
    if "去年" in cleaned:
        return now_year - 1, now_year - 1
    if "this year" in lowered:
        return now_year, now_year
    if "last year" in lowered:
        return now_year - 1, now_year - 1

    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", cleaned)]
    if not years:
        return None, None

    # 去重并排序
    unique_years = sorted(set(years))
    if len(unique_years) == 1:
        return unique_years[0], unique_years[0]
    return unique_years[0], unique_years[-1]

# ✅️
def _extract_conferences(text: str) -> List[str]:
    """从文本中提取会议简称，大小写不敏感。"""

    lowered = text.lower()
    result: List[str] = []
    for alias, canonical in _CONFERENCE_ALIASES.items():
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])" # ⚠️这里可能有点问题
        if re.search(pattern, lowered):
            result.append(canonical)
    return _deduplicate(result)

# ✅️
def _extract_topic(text: str) -> str:
    """抽取主题，优先匹配结构化语句，再做回退清洗。"""

    patterns = [
        r"关于\s*([^，。；;\n]+?)\s*(?:方面)?(?:的)?论文",
        r"(?:搜|搜索|检索|找|查找)\s*([^，。；;\n]+?)\s*(?:的)?论文",
        r"(?:主题|方向)\s*(?:是|为|:|：)\s*([^，。；;\n]+)",
        r"papers?\s+(?:about|on|regarding)\s+(.+?)(?=\s+(?:from|between|at|in)\b|[,.;\n]|$)",
        r"(?:search|find|look\s*for)\s+(?:papers?\s+)?(?:about|on|regarding)?\s*(.+?)(?=\s+(?:from|between|at|in)\b|[,.;\n]|$)",
        r"(?:topic|keyword|subject)\s*(?:is|:)?\s*(.+?)(?=[,.;\n]|$)",
    ]
    for pattern in patterns:
        matched = re.search(pattern, text, flags=re.IGNORECASE) # flags=re.IGNORECASE 使得正则匹配时忽略大小写，这样用户输入的“搜索 NLP 论文”或者“search nlp papers”都能被正确匹配到主题“NLP”；同时，正则模式中的\s*和非贪婪匹配.+?等设计，能够适应用户输入中可能存在的多余空格或其他文本，使得抽取更鲁棒；最后，正则模式中的(?=...)前瞻断言，能够确保我们抽取的主题后面跟着的是合理的上下文（比如年份、会议等），而不是一些无关的文本，从而提高抽取的准确性。
        if matched:
            return _clean_topic(matched.group(1))
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️⚠️⚠️⚠️看到这里了，明天继续加油⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
    # 回退到最近一句用户描述并剔除明显噪声词
    segments = [seg.strip() for seg in re.split(r"[。！？!\n]", text) if seg.strip()]
    if not segments:
        return ""

    candidate = ""
    for segment in reversed(segments): # 用了reversed，作用是：从后往前遍历这些文本片段，这样就能优先考虑用户最近的输入，因为在对话中，用户通常会在最后几条消息中明确表达他们的需求；同时，这样的设计也能更好地适应用户在对话过程中逐步明确参数的情况，比如用户可能先说“我想找一些关于NLP的论文”，然后又补充说“最好是近五年的”，这样的情况下，我们就能从后往前找到包含“关于NLP”的那条消息，并从中抽取出主题“NLP”，而不是被前面可能存在的其他文本干扰。
        if ("论文" in segment) or ("paper" in segment.lower()):
            candidate = segment
            break
    if not candidate:
        candidate = segments[-1] # 如果没有找到包含“论文”关键词的消息，就回退到最后一条消息，假设用户在最后一条消息中也可能描述了他们的需求；虽然这种回退策略可能会引入一些噪声，但在没有更明确线索的情况下，这也是一个合理的选择，能够保证我们至少有一个文本片段作为主题抽取的输入，而不是完全放弃抽取。
    candidate = re.sub(r"20\d{2}\s*年?", " ", candidate)
    candidate = re.sub(r"(?:最近|近)\s*[一二两三四五六七八九十\d]+\s*年", " ", candidate)
    candidate = re.sub(
        r"(?:from|between)\s+20\d{2}\s+(?:to|and)\s+20\d{2}",
        " ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"(?:last|past)\s*\d{1,2}\s*years?", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"(?:this|last)\s*year", " ", candidate, flags=re.IGNORECASE)
    for alias in _CONFERENCE_ALIASES.keys():
        candidate = re.sub(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            " ",
            candidate,
            flags=re.IGNORECASE,
        )

    for noisy in [
        "我想", "帮我", "请", "搜索", "检索", "查找", "找",
        "论文", "paper", "papers", "关于", "会议", "发表", "一下",
    ]:
        candidate = candidate.replace(noisy, " ")

    for noisy_en in [
        "search", "find", "look for", "look", "please", "paper",
        "papers", "about", "on", "regarding", "in", "at",
    ]:
        candidate = re.sub(rf"\b{re.escape(noisy_en)}\b", " ", candidate, flags=re.IGNORECASE) # re.escape(s) 就一件事：把字符串里的正则特殊字符加上反斜杠，让它们变成普通字面字符。

    return _clean_topic(candidate)

# ✅️
def _normalize_args(
    llm_result: Dict[str, Any],
    rule_result: Dict[str, Any],
    merged_text: str,
) -> ExtractArgsResult:
    """统一归一化输出，保证返回结构稳定。"""

    now_year = datetime.now().year

    # 优先使用 LLM 结果；如果 LLM 结果不完整，再用规则结果补全；如果两者都不行，最后回退到文本清洗的默认结果。
    topic = _clean_topic(llm_result.get("topic", ""))
    if not topic:
        topic = _clean_topic(rule_result.get("topic", ""))
    if not topic:
        topic = _clean_topic(merged_text)
    if not topic:
        topic = "未指定主题"

    # 也是先看看 LLM 结果里有没有合理的年份，如果没有再看规则抽取的结果；如果规则抽取也没有，再回退到默认值（当前年）。这里的合理性判断在 _needs_rule_fallback 里已经做过一次了，这里我们就直接按照优先级顺序来取值，最后再进行边界限制和合理性调整，确保最终返回的年份范围是合理的。
    start_year = _to_int(llm_result.get("start_year"))
    end_year = _to_int(llm_result.get("end_year"))
    if start_year is None and end_year is None:
        start_year = rule_result.get("start_year")
        end_year = rule_result.get("end_year")
    elif start_year is None:
        start_year = rule_result.get("start_year") or end_year
    elif end_year is None:
        end_year = rule_result.get("end_year") or start_year
    
    # 实在没招了，就这样保底 :)
    if start_year is None:
        start_year = now_year
    if end_year is None:
        end_year = now_year

    start_year = _clip_year(start_year, now_year)
    end_year = _clip_year(end_year, now_year)
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    conferences = _normalize_conferences(llm_result.get("conferences", []))
    if not conferences:
        conferences = _normalize_conferences(rule_result.get("conferences", []))
    if not conferences:
        conferences = DEFAULT_CONFERENCES.copy()

    return {
        "topic": topic,
        "start_year": start_year,
        "end_year": end_year,
        "conferences": conferences,
    }

# ✅️
def _normalize_conferences(raw_conferences) -> List[str]:
    """将会议输入归一化为简称列表并去重。"""

    if raw_conferences is None:
        return []

    values = _collect_conference_candidates(raw_conferences)
    # 此时的values已经是一个纯字符串列表了，接下来我们要对每个字符串进行清洗和映射，最终得到一个规范化的会议简称列表。
    normalized: List[str] = []
    for value in values:
        key = re.sub(r"[^a-z0-9]", "", value.lower())
        if key in _CONFERENCE_ALIASES_NORMALIZED:
            normalized.append(_CONFERENCE_ALIASES_NORMALIZED[key])
            continue

        # 兼容 `ACL EMNLP` 这类仅由简称组成的空格分隔输入。
        token_parts = re.split(r"\s+", value)
        if len(token_parts) > 1:
            token_normalized: List[str] = []
            all_mapped = True
            for part in token_parts:
                part_key = re.sub(r"[^a-z0-9]", "", part.lower())
                if part_key in _CONFERENCE_ALIASES_NORMALIZED:
                    token_normalized.append(_CONFERENCE_ALIASES_NORMALIZED[part_key]) # dcc_这里有问题❓⚠️
                else:
                    all_mapped = False
                    break
                
            # dcc_这里有问题❓⚠️
            if all_mapped and token_normalized: # 只有当所有部分都成功映射时，才将它们加入最终结果；如果有任何一个部分无法映射，我们就放弃这个整体，避免引入错误的会议名称。
                normalized.extend(token_normalized)
                continue

        normalized.append(value.upper() if value.isalpha() and len(value) <= 10 else value)

    return _deduplicate(normalized)

# ✅️
def _collect_conference_candidates(raw_conferences: Any) -> List[str]:
    """将会议字段拆分成候选项，尽量保留全称语义。"""

    values: List[str] = []
    split_pattern = r"[,，、/;；\n]+"

    if isinstance(raw_conferences, str):
        values = re.split(split_pattern, raw_conferences)
    elif isinstance(raw_conferences, list):
        for item in raw_conferences:
            if isinstance(item, str):
                values.extend(re.split(split_pattern, item))

    return [value.strip() for value in values if value and value.strip()]

# ✅️
def _clean_topic(topic: str) -> str:
    """清理主题两侧噪声字符和多余空格。"""

    topic = topic.strip()
    topic = re.sub(r"\s+", " ", topic)
    topic = topic.strip(" ，。；;!！?？")
    return topic

# ✅️
def _to_int(value) -> Optional[int]:
    """将字符串数字（含中文数字）转换为整数。"""

    if value is None:
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[text]
    return None

# ✅️
def _clip_year(year: int, now_year: int) -> int:
    """限制年份边界，避免不合理年份扰动流程。"""

    return max(YEAR_LOWER_BOUND, min(now_year + YEAR_UPPER_FUTURE_OFFSET, int(year)))

# ✅️
def _deduplicate(values: List[str]) -> List[str]:
    """保持顺序去重。"""

    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered