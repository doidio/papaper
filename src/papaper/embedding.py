# import hashlib
import os
import sys
from multiprocessing import Queue
from pathlib import Path

import tiktoken
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
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


def build(message: dict, log_q: Queue):
    try:
        load = message['load']
        embedding = message['embedding']
        cache = (Path(sys.executable).parent.parent / 'cache').as_posix()

        log_q.put('[EMBEDDING] initialize')

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

                log_q.put(f'[EMBEDDING] {parsed + 1} / {total} parsed {len(texts)} from {_.parent.name} {_.name}')
                parsed += 1

        log_q.put(f'[EMBEDDING] wait for building database')
        _ = FAISS.from_documents(docs, HuggingFaceEmbeddings(cache_folder=cache))

        log_q.put(f'[EMBEDDING] save to {embedding}')
        _.save_local(embedding)

        log_q.put(f'[EMBEDDING] COMPLETE')
    except Exception as e:
        log_q.put(f'[EMBEDDING] ERROR {e}')


def search(message: dict, log_q: Queue):
    try:
        query = message['query']
        embedding = message['embedding']
        cache = (Path(sys.executable).parent.parent / 'cache').as_posix()

        log_q.put('[EMBEDDING] load database')
        db = FAISS.load_local(embedding, HuggingFaceEmbeddings(cache_folder=cache))

        log_q.put('[EMBEDDING] search similar documents')
        docs = db.similarity_search(query, 100)
        docs = [(_.metadata.get('category'), _.metadata.get('title'), _.page_content) for _ in docs]
        log_q.put({'related documents': docs})

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
