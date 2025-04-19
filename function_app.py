# function_app.py
import azure.functions as func
from azure.functions import AuthLevel
from openai import OpenAI
import os
import json
import logging
import psycopg2
import uuid
from azure.storage.blob import BlobClient
from psycopg2.extras import RealDictCursor
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import chromadb

app = func.FunctionApp()

# Environment and secrets setup
keyVaultName = os.environ.get("KEY_VAULT_NAME")
KVUri = f"https://{keyVaultName}.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

DB_CONFIG = {
    "dbname": client.get_secret('PROJ-DB-NAME').value,
    "user": client.get_secret('PROJ-DB-USER').value,
    "password": client.get_secret('PROJ-DB-PASSWORD').value,
    "host": client.get_secret('PROJ-DB-HOST').value,
    "port": client.get_secret('PROJ-DB-PORT').value,
}

OPENAI_API_KEY = client.get_secret('PROJ-OPENAI-API-KEY').value
AZURE_STORAGE_SAS_URL = client.get_secret('PROJ-AZURE-STORAGE-SAS-URL').value
AZURE_STORAGE_CONTAINER = client.get_secret('PROJ-AZURE-STORAGE-CONTAINER').value
CHROMADB_HOST = client.get_secret('PROJ-CHROMADB-HOST').value
CHROMADB_PORT = client.get_secret('PROJ-CHROMADB-PORT').value

storage_resource_uri = AZURE_STORAGE_SAS_URL.split('?')[0]
token = AZURE_STORAGE_SAS_URL.split('?')[1]
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model = "gpt-3.5-turbo"

@app.function_name(name="chat")
@app.route(route="chat", auth_level=AuthLevel.ANONYMOUS)
def chat(req: func.HttpRequest) -> func.HttpResponse:
    stream = openai_client.chat.completions.create(
        model=model,
        messages=req.get_json()['messages'],
    )
    return func.HttpResponse(stream.choices[0].message.content)

@app.function_name(name="load_chat")
@app.route(route="load-chat", auth_level=AuthLevel.ANONYMOUS)
def load_chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        db = psycopg2.connect(**DB_CONFIG)
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT id, name, file_path, pdf_name, pdf_path, pdf_uuid FROM advanced_chats ORDER BY last_update DESC")
            rows = cursor.fetchall()

        records = []
        for row in rows:
            blob_sas_url = f"{storage_resource_uri}/{AZURE_STORAGE_CONTAINER}/{row['file_path']}?{token}"
            blob_client = BlobClient.from_blob_url(blob_sas_url)
            if blob_client.exists():
                blob_data = blob_client.download_blob().readall()
                messages = json.loads(blob_data)
                records.append({**row, "messages": messages})

        return func.HttpResponse(body=json.dumps(records), status_code=200)
    except Exception as e:
        logging.error(e)
        return func.HttpResponse(body=json.dumps({"detail": str(e)}), status_code=500)

@app.function_name(name="save_chat")
@app.route(route="save-chat", auth_level=AuthLevel.ANONYMOUS)
def save_chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        db = psycopg2.connect(**DB_CONFIG)
        data = req.get_json()
        chat_id, chat_name, messages, pdf_path, pdf_name, pdf_uuid = data.values()
        file_path = f"chat_logs/{chat_id}.json"

        blob_sas_url = f"{storage_resource_uri}/{AZURE_STORAGE_CONTAINER}/{file_path}?{token}"
        blob_client = BlobClient.from_blob_url(blob_sas_url)
        blob_client.upload_blob(json.dumps(messages, ensure_ascii=False, indent=4), overwrite=True)

        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO advanced_chats (id, name, file_path, last_update, pdf_path, pdf_name, pdf_uuid)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET name = EXCLUDED.name, file_path = EXCLUDED.file_path, last_update = CURRENT_TIMESTAMP, pdf_path = EXCLUDED.pdf_path, pdf_name = EXCLUDED.pdf_name, pdf_uuid = EXCLUDED.pdf_uuid
                """,
                (chat_id, chat_name, file_path, pdf_path, pdf_name, pdf_uuid)
            )
        db.commit()
        return func.HttpResponse(body=json.dumps({"message": "Chat saved successfully"}), status_code=200)
    except Exception as e:
        logging.error(e)
        return func.HttpResponse(body=json.dumps({"detail": str(e)}), status_code=500)

@app.function_name(name="delete_chat")
@app.route(route="delete-chat", auth_level=AuthLevel.ANONYMOUS)
def delete_chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        db = psycopg2.connect(**DB_CONFIG)
        chat_id = req.get_json()['chat_id']
        with db.cursor() as cursor:
            cursor.execute("SELECT file_path, pdf_path FROM advanced_chats WHERE id = %s", (chat_id,))
            result = cursor.fetchone()
            if not result:
                return func.HttpResponse(status_code=404, body=json.dumps({"detail": "Chat not found"}))
            file_path, pdf_path = result

        with db.cursor() as cursor:
            cursor.execute("DELETE FROM advanced_chats WHERE id = %s", (chat_id,))
        db.commit()

        for path in [file_path, pdf_path]:
            if path:
                blob_sas_url = f"{storage_resource_uri}/{AZURE_STORAGE_CONTAINER}/{path}?{token}"
                blob_client = BlobClient.from_blob_url(blob_sas_url)
                if blob_client.exists():
                    blob_client.delete_blob()

        return func.HttpResponse(body=json.dumps({"message": "Chat deleted successfully"}), status_code=200)
    except Exception as e:
        logging.error(e)
        return func.HttpResponse(body=json.dumps({"detail": str(e)}), status_code=500)

@app.function_name(name="upload_pdf")
@app.route(route="upload-pdf", auth_level=AuthLevel.ANONYMOUS)
def upload_pdf(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file = req.files.get("file")
        if file.content_type != "application/pdf":
            return func.HttpResponse(status_code=400, body=json.dumps({"detail": "Only PDF files are allowed."}))

        pdf_uuid = str(uuid.uuid4())
        file_path = f"pdf_store/{pdf_uuid}_{file.filename}"
        temp_path = f"/tmp/{file.filename}"

        with open(temp_path, "wb") as f:
            f.write(file.read())

        blob_sas_url = f"{storage_resource_uri}/{AZURE_STORAGE_CONTAINER}/{file_path}?{token}"
        BlobClient.from_blob_url(blob_sas_url).upload_blob(temp_path, overwrite=True)

        loader = PyPDFLoader(temp_path)
        texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

        embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        vectorstore = Chroma(client=chroma_client, collection_name="langchain", embedding_function=embedding_function)
        vectorstore.add_texts(
            [doc.page_content for doc in texts],
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=[{"pdf_uuid": pdf_uuid} for _ in texts]
        )

        os.remove(temp_path)

        return func.HttpResponse(body=json.dumps({"message": "File uploaded successfully", "pdf_path": file_path, "pdf_uuid": pdf_uuid}), status_code=200)
    except Exception as e:
        logging.error(e)
        return func.HttpResponse(body=json.dumps({"detail": str(e)}), status_code=500)

@app.function_name(name="rag_chat")
@app.route(route="rag-chat", auth_level=AuthLevel.ANONYMOUS)
def rag_chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        vectorstore = Chroma(client=chroma_client, collection_name="langchain", embedding_function=embedding_function)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"pdf_uuid": req.get_json()['pdf_uuid']}})

        llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question..."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks...\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain = create_retrieval_chain(
            create_history_aware_retriever(llm, retriever, contextualize_prompt),
            create_stuff_documents_chain(llm, qa_prompt)
        ).pick("answer")

        messages = req.get_json()['messages']
        chat_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in messages]

        response = rag_chain.invoke({"chat_history": chat_history[:-1], "input": chat_history[-1].content})
        return func.HttpResponse(response, status_code=200)
    except Exception as e:
        logging.error(e)
        return func.HttpResponse(body=json.dumps({"detail": str(e)}), status_code=500)
