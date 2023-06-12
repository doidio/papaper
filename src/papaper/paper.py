import json
import os
import re
import warnings
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path

from scholarly import scholarly
from scihub.util.download import SciHub


def main(message: dict, log_q: Queue):
    try:
        save_in = message['save_in']
        keyword = message['keyword']
        n_papers = message['n_papers']
        n_years = max(message['n_years'], 1) - 1

        save_in = Path(save_in)
        os.makedirs(save_in, exist_ok=True)

        metadata_json = save_in / f'{keyword}.json'
        if metadata_json.exists():
            metadata = json.loads(metadata_json.read_text(encoding='utf-8'))
        else:
            metadata = {}

        log_q.put('[PAPER] initialize')
        scihub = SciHub()
        scholar = scholarly.search_pubs(keyword, year_low=datetime.now().year - n_years, year_high=datetime.now().year)

        searched = sum([len(metadata[__]) for __ in metadata])
        log_q.put(f'{searched} / {n_papers}')

        while searched < n_papers:
            try:
                log_q.put('[PAPER] search next')
                paper = next(scholar)
                scholar.pub_parser.fill(paper)

                pub_year = paper['bib']['pub_year']
                title = paper['bib']['title']
                filename = re.sub(r'[\\/:*?"<>|]+', '', title)

                if pub_year not in metadata:
                    metadata[pub_year] = {}

                if title in metadata[pub_year]:
                    log_q.put(f'[PAPER] skip {pub_year} {title}')
                    metadata[pub_year][title].update(paper.copy())
                else:
                    metadata[pub_year][title] = paper.copy()

                    try:
                        log_q.put(f'[PAPER] try download {pub_year} {title}')
                        url = scihub.search(paper['pub_url'])
                        os.makedirs(save_in / pub_year, exist_ok=True)
                        scihub.download(url, save_in / pub_year, filename + '.pdf')
                        download = 'succeeded'
                    except Exception as e:
                        warnings.warn(f'SCIHUB {e}')
                        download = 'failed'

                    log_q.put(f'[PAPER] try download {download}')

                    metadata[pub_year][title]['download'] = download

                metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=4), encoding='utf-8')

                searched = sum([len(metadata[_]) for _ in metadata])
                log_q.put(f'[PAPER] {searched} / {n_papers}')
            except Exception as e:
                warnings.warn(f'SCHOLARLY {e}')

        log_q.put(f'[PAPER] COMPLETE')
    except Exception as e:
        log_q.put(f'[PAPER] ERROR: {e}')
