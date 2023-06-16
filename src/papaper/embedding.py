import hashlib
import os
from multiprocessing import Queue
from pathlib import Path

# import openai
from langchain.embeddings import HuggingFaceEmbeddings
from milvus import default_server
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tika import parser


def parse_file(filename: str, min_size=2000):
    parsed = parser.from_file(filename)
    content = parsed['content']

    content = content.split('\n')
    content_list = []
    count = 0

    for _ in content:
        if len(content_list) == 0:
            content_list.append([])

        if count > min_size:
            content_list.append([])
            count = 0

        content_list[-1].append(_)
        count += len(_)

    content_list = ['\n'.join(_) for _ in content_list]
    return content_list


# def embed(text: str):
#     embedding = openai.Embedding.create(input=[text], engine='text-embedding-ada-002')
#     embedding = embedding['data'][0]['embedding']
#     return embedding


def init():
    if utility.has_collection('paper'):
        return

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='Id', is_primary=True, auto_id=False),
        FieldSchema(name='sha1', dtype=DataType.VARCHAR, description='Content SHA1', max_length=40),
        FieldSchema(name='year', dtype=DataType.INT64, description='Paper year'),
        FieldSchema(name='title', dtype=DataType.VARCHAR, description='Paper title', max_length=200),
        FieldSchema(name='content', dtype=DataType.VARCHAR, description='Paper content', max_length=20000),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Content embedding', dim=768),
    ]
    schema = CollectionSchema(fields=fields, description='Paper collection')
    collection = Collection(name='paper', schema=schema)

    index_params = {
        'index_type': 'IVF_FLAT',
        'metric_type': 'L2',
        'params': {'nlist': 1024}
    }
    collection.create_index(field_name='embedding', index_params=index_params)


def build(message: dict, log_q: Queue):
    try:
        cache = message['cache']
        load = message['load']

        log_q.put('[EMBEDDING] initialize')
        connections.connect(host='127.0.0.1', port=default_server.listen_port)
        init()

        embeddings = HuggingFaceEmbeddings(cache_folder=cache)
        embed = embeddings.embed_query

        collection = Collection('paper')
        collection.load()
        collection.flush()
        n = collection.num_entities

        for subdir in os.listdir(load):
            subdir = Path(load) / subdir
            if not subdir.is_dir():
                continue

            for _ in os.listdir(subdir.as_posix()):
                _ = subdir / _

                log_q.put(f'[EMBEDDING] embed {_.parent.name} {_.name}')
                content_list = parse_file(_.as_posix())

                for text in content_list:
                    sha1 = hashlib.sha1(text.encode(encoding='utf-8')).hexdigest()
                    if len(collection.query(f'sha1 == "{sha1}"')) > 0:
                        continue
                    else:
                        collection.insert([[n], [sha1], [int(_.parent.name)], [_.name], [text], [embed(text)]])
                        n += 1

        log_q.put(f'[EMBEDDING] COMPLETE')
    except Exception as e:
        log_q.put(f'[EMBEDDING] ERROR: {e}')


def search(message: dict, log_q: Queue):
    try:
        cache = message['cache']
        text = message['text']

        log_q.put('[EMBEDDING] connect')
        connections.connect(host='127.0.0.1', port=default_server.listen_port)

        collection = Collection('paper')
        collection.load()
        collection.flush()

        embeddings = HuggingFaceEmbeddings(cache_folder=cache)
        embed = embeddings.embed_query

        results = collection.search(
            data=[embed(text)],
            anns_field='embedding',
            param={'metric_type': 'L2'},
            limit=10,
            output_fields=['year', 'title', 'content']
        )

        papers = []
        for _ in results[0]:
            papers.append((_.entity.get('year'), _.entity.get('title'), _.entity.get('content')))

        log_q.put({'related papers': papers})

        log_q.put(f'[EMBEDDING] COMPLETE')
    except Exception as e:
        log_q.put(f'[EMBEDDING] ERROR: {e}')
