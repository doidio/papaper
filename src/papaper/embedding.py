# import hashlib
import os
from multiprocessing import Queue
from pathlib import Path

import tiktoken
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from milvus import default_server
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tika import parser


def parse_file(filename: str):
    parsed = parser.from_file(filename)
    content = parsed['content']

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )

    texts = text_splitter.split_text(content)
    return texts


# def init():
#     if utility.has_collection('paper'):
#         return
#
#     fields = [
#         FieldSchema(name='id', dtype=DataType.INT64, descrition='Id', is_primary=True, auto_id=False),
#         FieldSchema(name='sha1', dtype=DataType.VARCHAR, description='SHA1', max_length=40),
#         FieldSchema(name='category', dtype=DataType.VARCHAR, description='Category', max_length=65535),
#         FieldSchema(name='title', dtype=DataType.VARCHAR, description='Title', max_length=65535),
#         FieldSchema(name='content', dtype=DataType.VARCHAR, description='Content', max_length=200),
#         FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding', dim=768),
#     ]
#     schema = CollectionSchema(fields=fields, description='Paper collection')
#     collection = Collection(name='paper', schema=schema)
#
#     index_params = {
#         'index_type': 'IVF_FLAT',
#         'metric_type': 'L2',
#         'params': {'nlist': 1024}
#     }
#     collection.create_index(field_name='embedding', index_params=index_params)


def build(message: dict, log_q: Queue):
    # try:
    cache = message['cache']
    load = message['load']
    embedding = message['embedding']

    log_q.put('[EMBEDDING] initialize')
    # connections.connect(host='127.0.0.1', port=default_server.listen_port)
    # init()

    # embed = embeddings.embed_query

    # collection = Collection('paper')
    # collection.load()
    # collection.flush()
    # ne = collection.num_entities

    total = 0
    for subdir in os.listdir(load):
        subdir = Path(load) / subdir
        if not subdir.is_dir():
            continue
        for _ in os.listdir(subdir.as_posix()):
            total += 1

    docs = []
    parsed = 0
    for subdir in os.listdir(load):
        subdir = Path(load) / subdir
        if not subdir.is_dir():
            continue

        for _ in os.listdir(subdir.as_posix()):
            _ = subdir / _

            texts = parse_file(_.as_posix())
            docs += [Document(page_content=t, metadata=dict(category=_.parent.name, title=_.name)) for t in texts]

            # for text in texts:
            #     sha1 = hashlib.sha1(text.encode(encoding='utf-8')).hexdigest()
            #     if len(collection.query(f'sha1 == "{sha1}"')) > 0:
            #         continue
            #     else:
            #         collection.insert([[ne], [sha1], [_.parent.name], [_.name], [text], [embed(text)]])
            #         ne += 1

            log_q.put(f'[EMBEDDING] {parsed + 1} / {total} parsed {len(texts)} from {_.parent.name} {_.name}')
            parsed += 1

    log_q.put(f'[EMBEDDING] wait for building database')
    _ = FAISS.from_documents(docs, HuggingFaceEmbeddings(cache_folder=cache))

    log_q.put(f'[EMBEDDING] save to {embedding}')
    _.save_local(embedding)

    log_q.put(f'[EMBEDDING] COMPLETE')


# except Exception as e:
#     log_q.put(f'[EMBEDDING] ERROR {e}')


def search(message: dict, log_q: Queue):
    try:
        cache = message['cache']
        query = message['query']
        embedding = message['embedding']

        log_q.put('[EMBEDDING] connect')
        # connections.connect(host='127.0.0.1', port=default_server.listen_port)

        # collection = Collection('paper')
        # collection.load()
        # collection.flush()

        # embeddings = HuggingFaceEmbeddings(cache_folder=cache)
        # embed = embeddings.embed_query

        # results = collection.search(
        #     data=[embed(text)],
        #     anns_field='embedding',
        #     param={'metric_type': 'L2'},
        #     limit=100,
        #     output_fields=['category', 'title', 'content']
        # )

        db = FAISS.load_local(embedding, HuggingFaceEmbeddings(cache_folder=cache))
        docs: list[Document] = db.similarity_search(query, 100)

        papers = []
        for _ in docs:
            papers.append((_.metadata.get('category'), _.metadata.get('title'), _.page_content))

        log_q.put({'related papers': papers})

        log_q.put(f'[EMBEDDING] COMPLETE')
    except Exception as e:
        log_q.put(f'[EMBEDDING] ERROR {e}')


def text_in_tokens(texts: list, tokens: int):
    if len(texts) > 0:
        enc = tiktoken.encoding_for_model('gpt-4')
        text = texts[0]
        n = len(enc.encode(text))

        if len(texts) > 1:
            for _ in texts[1:]:
                if n > tokens:
                    break

                text += '\n' + _
                n += len(enc.encode(_))

        return text, n
    return '', 0
