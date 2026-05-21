import os
import dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Dict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from utils._extract_args import _extract_args
from utils._search_title import _search_title
from utils._download import _download

dotenv.load_dotenv()

SAVE_PATH = "save/"

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    args: Dict  # 格式：{"topic": "xxx", "start_year": xxxx, "end_year": xxxx, "conferences": ["A", "B", ...]}
    results: List[Dict]  # 格式：[{"title": "xxx", "year": xxxx, "conference": "xxxx", "url": "xxxxx"}, ...]
    downloaded: set
    save_path: str

class Intent(BaseModel):
    intent: Literal["chat", "search"] = Field(default="chat", description="chat: 闲聊；search: 搜索论文")

class PaperSearchAgent:
    def __init__(self):
        self.app = self._get_graph()
        self.messages = []
        print("智能体初始化成功...")
    
    def _get_graph(self):
        graph = StateGraph(AgentState)
        llm = ChatDeepSeek(
            model = "deepseek-chat",
            max_tokens=2048,
            api_key=os.getenv("DEEPSEEK_KEY")
        )
        ID_llm = llm.model_copy()
        ID_llm = ID_llm.with_structured_output(Intent) # 意图识别Node使用结构化输出，让LLM只能输出"chat"或"search"，避免输出不规范文本导致的分支错误
        chat_llm = llm.model_copy() # 闲聊Node使用原始的ChatDeepSeek，保持其强大的对话能力

        def intent_detect(state: AgentState) -> AgentState:
            """
            意图识别Node
            让LLM判断用户要`闲聊（chat）`还是`搜索论文（search）`
            """
            system_prompt = SystemMessage(
                content="你是一名论文搜索智能体，你需要根据对话历史判断用户的意图是闲聊（chat）还是论文搜索（search）"
            )
            result = ID_llm.invoke([system_prompt] + state["messages"])
            print(f"用户意图为：{result.intent}")
            return {"intent": result.intent}
        
        def select_edge_by_intent(state: AgentState) -> AgentState:
            if state["intent"] is not None and state["intent"] == "search":
                return "search"
            return "chat"
        
        def chit_chat(state: AgentState) -> AgentState:
            """
            闲聊Node
            让LLM直接基于历史对话回答用户问题
            """
            response = chat_llm.invoke(state["messages"])
            return {"messages": [response]}
        
        def extract_args(state: AgentState) -> Dict:
            """
            参数抽取Node
            直接读取 state 中累计的历史对话，优先用当前 Graph 的 LLM 做结构化抽取；
            若抽取结果不完整或调用失败，再回退到 utils._extract_args 做兜底。
            该 Node 只返回对 state["args"] 的增量更新。
            """
            return _extract_args(state, llm)
        
        def confirm_topic(state: AgentState) -> AgentState:
            """
            主题确认Node
            向用户确认要搜索论文的主题
            TODO: 暂时跳过
            """
            args = state["args"]
            print(f"搜索主题为：{args["topic"]}")
            return {}
        
        def confirm_year(state: AgentState) -> AgentState:
            """
            年份确认Node
            向用户确认要搜索论文的年份范围
            TODO: 暂时跳过
            """
            args = state["args"]
            print(f"搜索年份为：{args["start_year"]}~{args["end_year"]}")
            return {}
        
        def confirm_conference(state: AgentState) -> AgentState:
            """
            会议确认Node
            向用户确认要搜索论文的发表会议
            TODO: 暂时跳过
            """
            args = state["args"]
            print(f"搜索会议为：{args["conferences"]}")
            return {}
        
        def search_title(state: AgentState) -> AgentState:
            """
            标题搜索Node
            使用浏览器搜索每个年份下各会议的Proceedings，并从中找到符合主题的论文
            TODO: 补全_search_title
            """
            return {"results": _search_title(state["args"])}

        def download(state: AgentState) -> AgentState:
            """
            论文下载Node
            基于论文结果列表下载开放论文 PDF。
            要求返回 set，供 summarize 节点直接判断下载状态。
            """
            return _download(state)
        
        def summarize(state: AgentState) -> AgentState:
            """
            论文搜索总结Node
            汇总下载到的论文，并汇报给用户
            """
            summary = state["results"] if state["results"] else [] 
            for i in range(len(summary)):
                id = summary[i]["id"]
                if id in state["downloaded"]:
                    summary[i]["downloaded"] = True
                else:
                    summary[i]["downloaded"] = False
            prompt = f"""基于所给的论文信息（Python列表格式），将其转化成自然语言汇报给用户。如果列表为空，则直接回答“没有找到任何符合条件的论文”
            论文列表：
            {summary}"""
            response = llm.invoke([HumanMessage(content=prompt)])
            return {"messages": [response]}


        graph.add_node("intent_detect", intent_detect)
        graph.add_edge(START, "intent_detect")
        graph.add_conditional_edges(
            "intent_detect",
            select_edge_by_intent,
            {
                "chat": "chit_chat",
                "search": "extract_args",
            }
        )

        graph.add_node("chit_chat", chit_chat)
        graph.add_edge("chit_chat", END)

        graph.add_node("extract_args", extract_args)
        graph.add_edge("extract_args", "confirm_topic")

        graph.add_node("confirm_topic", confirm_topic)
        graph.add_edge("confirm_topic", "confirm_year")

        graph.add_node("confirm_year", confirm_year)
        graph.add_edge("confirm_year", "confirm_conference")

        graph.add_node("confirm_conference", confirm_conference)
        graph.add_edge("confirm_conference", "search_title")

        graph.add_node("search_title", search_title)
        graph.add_edge("search_title", "download")

        graph.add_node("download", download)
        graph.add_edge("download", "summarize")

        graph.add_node("summarize", summarize)
        graph.add_edge("summarize", END)

        return graph.compile()

    def invoke(self, message: str):
        self.messages.append(HumanMessage(content=message))
        result = self.app.invoke({"messages": self.messages, "save_path": SAVE_PATH})
        self.messages = result["messages"]
        return result["messages"][-1]


if __name__ == "__main__":
    agent = PaperSearchAgent()
    while True:
        question = input()
        response = agent.invoke(question)
        print(response.content)
