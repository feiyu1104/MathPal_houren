# readme

## 项目介绍

这是一个基于 LangGraph 的小学数学AI伴学系统，可以实现讲解、出题、判断对错、分析学情等功能。

### 系统结构

本项目采用主管代理，整体架构为：![架构图](img\架构图.png)

整体流程为：init → supervisor → {explain | generate | grade | report | fallback} → polish → user_profile → END

### 节点介绍

- supervisor节点接受用户输入，进行意图分析以及知识点提取

- question_generating_agent节点可以根据当前的知识点动态生成一道数学选择题

- explain_knowledge_agent节点可以讲解当前知识点

- grade_agent节点可以判断用户回答的对错

- user_profile_agent节点可以绘制用户的画像

- ###### build_study_report：总结所有已学知识点的掌握情况

- fallback_agent：异常输入处理，在非数学话题时提醒用户回到数学场景

- polish _agent：结合学生喜欢的语言风格和学习兴趣进行回答优化

---

## 快速使用

### API申请

你需要申请一个[Qwen](https://bailian.console.aliyun.com/&tab=doc?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.4.4ad67b08WPVLoI&tab=doc#/doc/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2840915.html&renderType=iframe) API以及一个[LangSmith](https://smith.langchain.com/settings) 的 API 密钥 

### 项目部署

复制以下命令到终端来下载代码

```shell
# 项目部署
git clone https://gitee.com/xuefeiyuy/langgraph_my.git
# 创建虚拟环境
conda create -n 环境名字 python=3.11
conda activate 环境名字
# 下载依赖
pip install -r requirements.txt
```

安装LangGraph CLI

输入

```shell
# Python >= 3.11 is required.
pip install --upgrade "langgraph-cli[inmem]"
```

### 创建 LangGraph 应用程序 

输入以下命令，创建一个new-langgraph-project-python模板

```
langgraph new path/to/your/app --template new-langgraph-project-python
```

打开新 LangGraph 应用的根目录中，输入以下命令安装依赖项

```shell
pip install -e . 
```

用下载代码中的.env文件替换新 LangGraph 应用文件根目录下的.env，将你申请的两个api填入相应的位置

用下载代码中的graph.py文件替换新 LangGraph 应用文件中src/agent/graph.py文件

将下载代码中的vectorstore文件（向量库）和primary_math_docs文件（五年级上册知识点总结）复制到LangGraph 应用文件根目录下

### 运行LangGraph Server 

输入

```shell
langgraph dev
```

即可自动打开网页，你可以在这里输入消息，查看系统运行

![](img\LangGraph Server.png)

---

## 示例运行

输入”你是谁“

![fallback.png](img\fallback.png)

输入”你能讲一讲分数加法吗“

![explain](img\explain.png)

输入”出一道题“

![question](img\question.png)

输入”答案是B“

![grade](img\grade.png)

输入”我学的怎么样“

![study](img\study.png)

---

## 项目注意

项目全部使用Qwen模型，你可以进行替换

项目使用的向量库是基于小学五年级上册知识点，应用于explain_knowledge_agent，你可以进行替换

1. 将txt文件放置于primary_math_docs
2. 修改vector_make.py文件中的相应地址
3. 运行vector_make.py
4. 修改graph.py文件中的地址

项目只提供核心代码，没有测试代码，只能使用local server来测试

---

感谢您的使用！

