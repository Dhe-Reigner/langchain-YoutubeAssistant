from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# video_url = "https://youtu.be/fgImDCSp-VI"
def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_document(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    # return db
    return docs

def get_response_from_query(db, query, k=4):
    # text-davinci can handle 4097 tokens
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = "".join([d.page_content for d in docs])
    
    llm = OpenAI(model="text-davinci-003")
    
    prompt = PromptTemplate(
        input_variables=["question",docs],
        template="What is the following information about {question}? \n\nAvailable context:\n{docs}",
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response  =response.replace("\n", "")
    return response