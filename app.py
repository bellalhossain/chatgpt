import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

#try to use my own script
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone

os.environ['OPENAI_API_KEY'] ='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'

# App framework
st.title('🦜🔗 LangChain ChatGPT')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a script title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

loader = TextLoader('state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
 
# initialize pinecone
pinecone.init(
    api_key="860c5621-0910-438e-9eb6-30b50dd6cff7",
    environment="asia-southeast1-gcp-free"
)

index_name = "langchain-demo"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536,metric="cosine", pods=1, pod_type="p1.x1")
    print("Index has been created in pinecone")
else:
    print("Index already exist there")


docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
print(docs)

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    print(title)
    wiki_research = docsearch.similarity_search(title) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)
        

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)