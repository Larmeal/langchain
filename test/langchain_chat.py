from langchain_core.prompts import (
    HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
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

memory = ConversationBufferMemory(
    memory_key="messages",
    return_messages=True
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
    memory=memory
)

while True:
    content = input(">> ")

    output = chain.invoke(
        {
            "content": content
        }
    )

    print(output)