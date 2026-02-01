"""LangGraph single-node graph template.
Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import os
import random
import time
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from openai import OpenAI

random.seed(int.from_bytes(os.urandom(4), "big") ^ int(time.time() * 1e6))
# 加载 .env 文件，加载api
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_API_BASE")


def _safe_last_user_message(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    return messages[-1].content


class CustomState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_topic: str  # 当前知识点
    all_topics: list  # 已经学习过的知识点
    preferred_teaching_style: str  # 偏好的教学方式
    next_node: str  # 下一个要跳转的节点名
    response: str  # 模型回答
    preferred_language_style: str  # 偏好的语言风格   
    learning_interesting: str  # 学习兴趣
    study: list[dict]  # 学习情况


class DashscopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            r = self.client.embeddings.create(
                model="text-embedding-v4",
                input=t,
                dimensions=1024,
                encoding_format="float"
            )
            res.append(r.data[0].embedding)
        return res

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def init_or_continue(state: CustomState) -> Command[Literal["supervisor"]]:
    """第一次运行把空值补齐，再交回 supervisor 继续"""
    defaults = {
        "messages": [],
        "current_topic": "无",
        "all_topics": [],
        "preferred_teaching_style": "直接输出",
        "preferred_language_style": "有趣",
        "learning_interesting": "一般",
        "next_node": "",
        "response": "",
        "study": [],
    }
    # 只补充缺失的 key
    patched = {**defaults, **state}
    if state.get("messages") is not None and state.get("current_topic") is not None:
        return Command(goto="supervisor", update=state)
    return Command(goto="supervisor", update=patched)


def supervisor(state: Annotated[CustomState, InjectedState]) -> CustomState:
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0.5,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_API_BASE
    )
    messages = state.get("messages", [])
    # 倒数第一条：用户的话
    user_msg = _safe_last_user_message(messages)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是专业的意图识别助手，请根据用户的输入内容：{user_data}，完成以下任务：
        1. 判断其当前意图，可能包括：
        - 讲解数学知识点
        - 出题检验
        - 查询学习状态
        - 比对答案
        - 其它
        2. 请从用户输入中提取具体涉及的数学知识点名称
        要求：
        - 只需输出对应的意图名称即可（例如：“出题检验”）
        - 一定不要添加任何解释或额外内容
        - 知识点的提取必须基于用户输入内容，如果没有提取到，就输出：意图|无
        - 当用户的输入包含类似答案信息(例如用户输入含有A、B、C、D或者“答案”“第几个”这种字样)时，选择比对答案
        - 如果判断用户意图为出题检验，并且没有明确说明数学知识点，直接输出：出题检验|无
        - 对于查询学习状态与比对答案，直接输出：意图|无
        - 当判断意图不属于讲解数学知识点（知识点严格属于数学）、出题检验、查询学习状态、比对答案，则认为当前意图为其它，输出：其它|无
        请绝对严格按照如下格式输出（不要添加解释）：
        意图|知识点
        示例输出：
        讲解知识点|分数加减法
        """)
    ])
    chain = prompt | llm | StrOutputParser()
    out_data = chain.invoke({"user_data": user_msg})
    # print("目前意图识别智能体输出："+data)
    parts = out_data.split('|', 1)
    intent = parts[0].strip() if parts else "其它"
    part = parts[1].strip() if len(parts) > 1 else "无"
    if part == "无":
        topic = state.get("current_topic", "")
    else:
        topic = part
    next_map = {
        "讲解知识点": "explain_knowledge_agent",
        "出题检验": "question_generating_agent",
        "比对答案": "grade_agent",
        "查询学习状态": "build_study_report",
        "其它": "fallback_agent"
    }
    next_node = next_map.get(intent, "fallback_agent")  # 如果键不存在，就返回默认值 "fallback_agent"

    all_topics = state.get("all_topics", [])
    return {
        "current_topic": topic,
        "all_topics": all_topics + ([topic] if topic and topic not in all_topics else []),
        "next_node": next_node
    }


def explain_knowledge_agent(state: Annotated[CustomState, InjectedState]) -> CustomState:
    # 初始化语言模型
    llm = ChatOpenAI(
        openai_api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_BASE,
        model="qwen-plus-2025-04-28"
    )
    topic = state.get("current_topic", "")
    # 检索相关上下文
    # —— 新增：先检索 ——
    docs = retriever.get_relevant_documents(topic)
    context_text = "\n".join(doc.page_content for doc in docs) or "暂无相关资料。"
    # 提示模板（用于知识讲解）
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位经验丰富的小学数学老师，请根据用户的问题以及下面信息，直接向用户讲解以下知识点{topic}：
        目前用户已掌握知识点：{master_topics}
        参考内容：{context}
        要求：
        - 解释清晰易懂
        - 只举一个例子说明
        - 不要使用 markdown 格式
        - 输出简洁明了
        - 不要输出与讲解无关的特殊字符
        """),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm | StrOutputParser()
    # 调用模型并获取结果
    input_data = {
        "topic": topic,
        "context": context_text,
        "master_topics": state.get("all_topics", ""),
        "messages": state.get("messages", [])  # 对话历史消息列表
    }
    response = chain.invoke(input_data)
    updated_state = {
        "response": response,
    }
    return updated_state


def question_generating_agent(state: CustomState) -> CustomState:
    # 1. 准备多套文案
    OPENING_LINES = [
        "我出的题是：",
        "来挑战一下这道题吧！请做：",
        "脑力小测来啦——",
        "这道题能难倒你吗？请做：",
        "准备好答题了吗？请做：",
        "叮咚！今日份烧脑已送达：",
        "小试牛刀，请接招：",
        "学霸检测题上线！请做：",
        "别眨眼，题目来了：",
        "高能预警，题目出没！请做：",
        "答题时间到！请做：",
        "脑筋急转弯？不，是数学转弯：",
        "是时候展现真正的技术了：",
        "这道题据说只有 3% 的人能秒答：",
        "数学小火车呜呜呜，开过来啦："
    ]

    CLOSING_LINES = [
        "只需要说出正确答案的选项字母就OK~",
        "只需给出正确选项字母即可~",
        "直接回复 A/B/C/D 就行啦！",
        "不用解释，甩个字母过来！",
        "答案就藏在 A/B/C/D 中，选一个吧！",
    ]

    # 使用 DashScope 提供的兼容 OpenAI 接口服务
    llm = ChatOpenAI(
        model="qwen-plus-2025-04-28",
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_API_BASE
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位经验丰富的数学老师，请根据以下数学知识点：{topic} 生成一个小学数学选择题。

        输出格式必须严格按照如下格式：
        <题目>|<选项A>,<选项B>,<选项C>,<选项D>
        示例输出：
        1+1等于几|1,2,3,4

        要求：
        - 严格依据知识点出题
        - 只输出一道题
        - 不要任何额外说明或格式
        - 避免使用Mardown形式
        - 选项中不需要包含ABCD这四个表示选项的符号
        - 问题的答案随机分布，不要一直是同一个选项
        - 所出问题要保证问题的正确性以及选项中存在正确答案
        """)
    ])
    chain = prompt | llm | StrOutputParser()
    input_data = {"topic": state.get("current_topic", "")}
    raw = chain.invoke(input_data)
    # 解析输出 
    parts = raw.strip().split('|')
    if len(parts) != 2:
        return {
            "messages": [AIMessage(content="题目生成失败，请再试一次。")]
        }
    question, opts = parts
    options = opts.split(',')
    if len(options) < 4:
        return {
            "messages": [AIMessage(content="题目生成失败，请再试一次。")]
        }
    opening = random.choice(OPENING_LINES)
    closing = random.choice(CLOSING_LINES)
    q_text = (
        f"{opening}{question}\n"
        f"A) {options[0]}  B) {options[1]}  C) {options[2]}  D) {options[3]}\n"
        f"小贴士：{closing}"
    )
    updated_state = {
        "messages": [AIMessage(content=q_text)]  # 会自动追加
    }
    return updated_state


def grade_agent(state: CustomState) -> CustomState:
    llm = ChatOpenAI(
        model="qwen-plus-2025-04-28",
        openai_api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_BASE
    )

    messages = state.get("messages", [])
    if len(messages) < 2:
        return {"response": "请先出题并作答后再进行判题。"}
    # 倒数第二条：AI 出的题目
    question_msg = messages[-2].content
    # 倒数第一条：用户的回答
    user_msg = messages[-1].content

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是一位小学数学老师。请根据【题目】：{question}与【学生回答】：{answer}完成以下任务：
        1. 判断学生回答是否正确（正确 / 错误）。
        2. 给出正确答案及简要解析。
        输出格式：
        判断|正确答案|解析
        
        要求：
        - 严格按照输出格式输出，不要任何额外说明或格式
        - 正确答案应只包含选项字母
        - 解析应简洁明了，不出现与讲解无关的特殊字符
        
         示例输出：
        正确|C|因为 2/3 = 4/6 所以选 C
        """)
    ])
    # 直接传递最后两条消息的内容
    raw = (prompt | llm | StrOutputParser()).invoke({"question": question_msg, "answer": user_msg})
    parts = raw.strip().split('|')
    if len(parts) != 3:
        parts = ["错误", "未知", "解析暂缺"]
    result, correct_ans, explanation = parts
    is_correct = result.strip() == "正确"
    response_text = (
        f"回答正确！解析：{explanation.strip()}" if is_correct
        else f"答错了，正确答案是 {correct_ans.strip()}，解析：{explanation.strip()}"
    )
    return {
        "response": response_text
    }


def user_profile_agent(state: CustomState) -> CustomState:
    llm = ChatOpenAI(
        openai_api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_BASE,
        model="qwen-plus")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的学习行为分析师，专注于分析小学五年级学生的数学学习行为。
    请根据对话内容以及所有提问的知识点{all_topics}，完成以下分析任务：
    1. 根据所有提问的知识点，逐个分析，若有问答涉及某知识点，则统计其对应的题目数与正确率（正确题数 / 总题数），注意不要有重复；
    2. 对每个知识点判断是否掌握（掌握/没有掌握），若相关题目正确率 >50% 视为掌握；
    3. 分析用户偏好的语言风格（有趣/严肃）；
    4. 分析用户偏好的交流方式（直接输出/引导式提问）；
    5. 分析用户的学习兴趣（较差/一般/优秀），依据对话活跃度与问题深度；
    输出格式：
    知识点1/掌握情况/题目数/正确率@知识点2/掌握情况/题目数/正确率|语言风格|交流方式|学习兴趣
    输出示例：
    分数加减法/没有掌握/8/50%@方程式/掌握/2/100%|有趣|引导式提问|一般
    要求：
    - 所有分析必须基于对话内容，不要引入外部假设；
    - 知识点与知识点之间严格以@划分；
    - 正确率高于50%视为掌握；
    - 若无相关题目，则对应字段填写0和0%；
    - 无法推断时，优先选择“一般”或“直接输出”作为默认值；
    - 严格按照格式输出，不得添加任何额外文本；
    """), MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm | StrOutputParser()
    input_data = {
        "messages": state.get("messages", []),  # 对话历史消息列表
        "all_topics": state.get("all_topics", ""),
    }
    parts = chain.invoke(input_data)
    part_list = parts.split('|')
    if len(part_list) != 4:
        knowledge_part, language_style, interaction_style, learning_interest = "", "一般", "直接输出", "一般"
    else:
        knowledge_part, language_style, interaction_style, learning_interest = part_list

    # 分割知识点信息
    knowledge_parts = knowledge_part.split('@') if knowledge_part else []

    study_topics = []
    for kp in knowledge_parts:
        if not kp:
            continue
        # 假设格式为 知识点/掌握情况/题目数/正确率
        topic_data = kp.split('/')
        if len(topic_data) != 4:
            continue
        topic_name = topic_data[0].strip()  # 提取知识点名称
        control_situation = topic_data[1].strip()  # 提取掌握情况
        question_num = topic_data[2].strip()  # 提取题目数
        question_acc = topic_data[3].strip()  # 提取正确率

        study_topics.append({
            "name": topic_name,
            "control_situation": control_situation,
            "question_num": question_num,
            "question_acc": question_acc
        })

    return {
        "study": study_topics,
        "preferred_teaching_style": interaction_style.strip(),
        "preferred_language_style": language_style.strip(),
        "learning_interesting": learning_interest.strip()
    }


def build_study_report(state: CustomState) -> CustomState:
    """
    根据 study 列表生成一句话学习情况汇报
    """
    study = state.get("study", [])
    if not study:
        return {"response": "暂无学习记录。"}

    learned = []
    for item in study:
        name = item["name"]
        acc = item["question_acc"]
        num = item["question_num"]
        status = "已掌握" if item["control_situation"] == "掌握" else "待加强"
        learned.append(f"针对{name}，已出题{num}道，正确率{acc}，{status}")
    text = f"到目前为止，{'; '.join(learned)}。"
    return {
        "response": text
    }


def fallback_agent(state: CustomState) -> CustomState:
    llm = ChatOpenAI(
        openai_api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_BASE,
        model="qwen-plus"
    )
    messages = state.get("messages", [])
    # 倒数第一条：用户的回答
    user_msg = _safe_last_user_message(messages)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位很擅长解决问题的数学智能助手-小仁，针对用户输入内容：{data}，帮助用户解决问题。
        要求：
        - 输出简短
        - 不要输出与回答无关的特殊字符
        - 输出以段落为主，可以进行分段，但是不要分点
        - 如果用户问题与学习无关，在简单回答问题之后，提醒用户自己是一个数学智能助手-小仁，只具有讲解数学知识点、出数学题、分析学情的功能，让其重新提问
    """)
    ])
    chain = prompt | llm | StrOutputParser()

    input_data = {
        "data": user_msg
    }
    out_data = chain.invoke(input_data)
    updated_state = {
        "response": out_data  # 覆盖为优化后的文本
    }
    return updated_state


def polish_agent(state: CustomState) -> CustomState:
    llm = ChatOpenAI(
        openai_api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_BASE,
        model="qwen-plus"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位优秀的语言学家。请将输入的文本在保留原意的前提下转化为特定语言风格的文本，你有50%概率需要根据用户目前的学习兴趣增加相应内容。
         输入文本：{response}
         特定语言风格：{preferred_language_style}
         用户目前的学习兴趣：{learning_interesting}
        要求：
        - 在转化语言风格时，严格保留输入文本原意
        - 如果需要根据用户目前的学习兴趣增加相应内容，则当用户学习兴趣较差或一般时鼓励用户，兴趣较好时夸赞用户
        - 输出简洁明了
        - 不要输出与讲解无关的特殊字符
        - 输出以段落为主，可以进行分段，但是不要分点
    """)
    ])
    chain = prompt | llm | StrOutputParser()

    input_data = {
        "response": state.get("response", ""),
        "preferred_language_style": state.get("preferred_language_style", ""),
        "learning_interesting": state.get("learning_interesting", ""),
    }
    polished = chain.invoke(input_data)

    updated_state = {
        "messages": [AIMessage(content=polished)],  # 会自动追加
        "response": polished  # 覆盖为优化后的文本
    }
    return updated_state


# 初始化向量库
# 加载向量库
persist_dir = "vectorstore/grade5"
embed = DashscopeEmbeddings(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_BASE
)
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embed)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})   # 取最相关 2 段
# 初始化状态图
graph = StateGraph(CustomState)
# 添加节点
graph.add_node("init", init_or_continue)
graph.add_node("supervisor", supervisor)  # 监控节点
graph.add_node("explain_knowledge_agent", explain_knowledge_agent)  # 知识点讲解节点
graph.add_node("question_generating_agent", question_generating_agent)  # 出题节点
graph.add_node("grade_agent", grade_agent)  # 评分节点
graph.add_node("polish_agent", polish_agent)  # 优化节点
graph.add_node("build_study_report", build_study_report)  # 学习报告节点
graph.add_node("user_profile_agent", user_profile_agent)  # 用户画像节点
graph.add_node("fallback_agent", fallback_agent)  # 其它节点

# 添加边
# 从初始节点到监控节点
graph.add_edge(START, "init")  # 指向 init
graph.add_edge("init", "supervisor")  # init 之后到 supervisor


# 根据用户意图跳转到不同节点
# 从 supervisor 出发的多分支
def route_after_supervisor(state: CustomState) -> Literal[
    "explain_knowledge_agent",
    "question_generating_agent",
    "grade_agent",
    "build_study_report",
    "fallback_agent"
]:
    return state["next_node"]


graph.add_conditional_edges("supervisor", route_after_supervisor)

# 连边
graph.add_edge("explain_knowledge_agent", "polish_agent")
graph.add_edge("grade_agent", "polish_agent")
graph.add_edge("build_study_report", "polish_agent")
graph.add_edge("fallback_agent", "polish_agent")
graph.add_edge("polish_agent", "user_profile_agent")
graph.add_edge("user_profile_agent", END)
graph.add_edge("question_generating_agent", END)

# 设置保存器
saver = InMemorySaver()
app = graph.compile(checkpointer=saver)

# 打印状态图
# print(graph)

