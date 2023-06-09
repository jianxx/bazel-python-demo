from langchain.document_loaders import GitLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)


class GolangCodeTextSplitter(CharacterTextSplitter):
    """Attempts to split the text along Golang syntax."""

    def __init__(self, **kwargs: any):
        """Initialize a GolangCodeTextSplitter."""
        # separators = [
        #     # First, try to split along class definitions
        #     "\nstruct ",
        #     "\nfunc ",
        #     "\n\tfunc ",
        #     # Now split by the normal type of lines
        #     "\n\n",
        #     "\n",
        #     " ",
        #     "",
        # ]
        super().__init__(**kwargs)


def golang_file_filter(filename: str) -> bool:
    """Filter out non-golang files."""
    return filename.endswith(".go")


def code_summary():
    # 加载 git 项目
    loader = GitLoader(repo_path='/Users/sean/workspace/dapr/dapr', branch='master', file_filter=golang_file_filter)
    documents = loader.load()

    # 初始化Golang代码分割器
    golang_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # 分割代码
    split_documents = golang_splitter.split_documents(documents)
    for document in split_documents:
        print("document: "+str(document))
    print("split documents: "+str(len(split_documents)))

    # 初始化openai embeddings
    embeddings = OpenAIEmbeddings()

    # 将数据存入向量存储
    vector_store = Chroma.from_documents(split_documents, embeddings)

    retriever = vector_store.as_retriever()
    print("retriever: "+str(retriever))

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
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name='gpt-3.5-turbo'), retriever, condense_question_prompt=prompt)

    chat_history = []
    question = '这个代码仓库实现了什么功能？'
    # 开始发送问题 chat_history 为必须参数,用于存储对话历史
    result = qa({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    print(result['answer'])
