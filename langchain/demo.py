from langchain.agents import AgentType, load_tools, initialize_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def first_qa():
    llm = OpenAI(model_name="text-davinci-003", max_tokens=1024)
    print(llm("怎么评价人工智能"))


def google_search():
    # 加载 OpenAI 模型
    llm = OpenAI(temperature=0, max_tokens=2048)

    # 加载 serpapi 工具
    tools = load_tools(["serpapi"])
    # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # 运行 agent
    agent.run("What's the date today? What great events have taken place today in history?")


def youtube_analyze():
    # 加载 youtube 频道
    loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=Dj60HHy-Kqk', add_video_info=True)
    # 将数据转成 document
    documents = loader.load()

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )

    # 分割 youtube documents
    documents = text_splitter.split_documents(documents)
    for document in documents:
        print("document: "+str(document))
    print("youtube documents: "+str(len(documents)))

    # 初始化 openai embeddings
    embeddings = OpenAIEmbeddings()
    print("embeddings: "+str(embeddings))

    # 将数据存入向量存储
    vector_store = Chroma.from_documents(documents, embeddings)
    # 通过向量存储初始化检索器
    retriever = vector_store.as_retriever()

    system_template = """
    Use the following context to answer the user's question.
    If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
    -----------
    {context}
    -----------
    {chat_history}
    """

    # 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}')
    ]

    # 初始化 prompt 对象
    prompt = ChatPromptTemplate.from_messages(messages)

    # 初始化问答链
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.1, max_tokens=2048), retriever, prompt)

    chat_history = []
    question = input('问题：')
    # 开始发送问题 chat_history 为必须参数,用于存储对话历史
    result = qa({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    print(result['answer'])
