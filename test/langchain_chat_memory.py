from langchain_core.prompts import (
    HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from langchain.chains.llm import LLMChain
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="llama3")


######################################## NEW VERSIONS #########################################
# memory = ChatMessageHistory(
#     memory_key="messages"
# )

# prompt = ChatPromptTemplate(
#     input_variables=["content", "messages"],
#     messages=[
#         MessagesPlaceholder(
#             variable_name="messages"
#         ),
#         HumanMessagePromptTemplate.from_template(
#             "{content}"
#         ),
#     ],
# )

# chain = prompt | llm

# chain_with_memory = RunnableWithMessageHistory(
#     runnable=chain,
#     get_session_history=lambda session_id: memory,
#     input_messages_key="content",
#     history_messages_key="messages"
# )

# while True:
#     content = input(">> ")

#     output = chain_with_memory.invoke(
#         input={
#             "content": content
#         },
#         config={
#             "configurable": {"session_id": "test"}
#         }
#     )

#     print(output)

#################################################################################################
######################################## OLD VERSIONS ###########################################

# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("test/src/chat_history_crane.json"),
#     memory_key="messages",
#     return_messages=True
# )

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=llm
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(
            variable_name="messages"
        ),
        HumanMessagePromptTemplate.from_template(
            "{content}"
        ),
    ],
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

while True:
    content = input(">> ")

    output = chain.invoke(
        {
            "content": content
        }
    )

    print(output["text"])