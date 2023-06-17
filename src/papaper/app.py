import atexit
import json
import os
import threading
from multiprocessing import Queue, Process, freeze_support
from pathlib import Path
from queue import Empty

import flet as ft
import psutil
from milvus import default_server

from papaper import paper, embedding


class App:
    def __init__(self):
        self.page: ft.Page

        self.log_q = Queue()

        self.paper_p = None
        self.paper_in = None

        self.embedding_build_p = None
        self.embedding_build_in = None

        self.embedding_search_p = None
        self.embedding_search_in = None

        self.config_path = Path(os.getenv('APPDATA')) / 'Papaper' / 'config.json'

        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
        else:
            self.config = {}

        self.related_texts = []

    def save_config(self, **kwargs):
        self.config.update(kwargs)
        os.makedirs(self.config_path.parent, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config, indent=4, ensure_ascii=False), encoding='utf-8')

    def __call__(self, page: ft.Page):
        self.page = page

        self.page.title = 'Papaper'
        self.page.padding = 20

        def close_dialog(_):
            self.page.dialog.open = False
            self.page.update()

        self.page.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(''),
            content=ft.Text(''),
            actions=[ft.TextButton('Ok', on_click=close_dialog), ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: print('Modal dialog dismissed!'),
        )

        # tab 1
        self.config_tab = ft.Column()

        def on_help(_):
            self.page.snack_bar = ft.SnackBar(ft.TextField(value='''
1. 先确认当前网络可以访问https://scholar.google.com/
2. 先确认已安装Java运行时https://www.java.com/download/
3. 设置Save路径到你的存档目录，其中将会保存下载的论文和用于检索的向量数据库
4. 数据库启动大约1分钟后才能用于构建和检索，更改Save路径之后必须Restart数据库服务
5. 论文检索、下载、数据库构建、相似性搜索，都不会消耗付费资源
                    ''', read_only=True, multiline=True, border=ft.InputBorder.NONE, color='white'), duration=30000)
            self.page.snack_bar.open = True
            self.page.update()

        self.config_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.ElevatedButton('README', expand=1, on_click=on_help))

        self.config_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        self.config_tab.controls.append(bar := ft.Row())

        _ = Path.home() / 'Desktop' / 'papaper' / 'save'
        _ = _.absolute().as_posix()
        self.save_ui = ft.TextField(label='Save', value=self.config.get('save', _), expand=1,
                                    on_change=lambda e: self.save_config(save=e.control.value))
        bar.controls.append(self.save_ui)

        def on_restart_server(_):
            default_server.stop()
            default_server.set_base_dir((Path(self.save_ui.value) / 'embedding').as_posix())
            default_server.start()

        bar.controls.append(ft.ElevatedButton('Restart', on_click=on_restart_server))

        on_restart_server(None)

        def on_open_save(_):
            os.startfile(self.save_ui.value)

        bar.controls.append(ft.ElevatedButton('Open', on_click=on_open_save))

        self.config_tab.controls.append(bar := ft.Row())

        self.kill_service_ui = ft.Checkbox(label='exit service when close', expand=1,
                                           value=self.config.get('kill_service', False),
                                           on_change=lambda e: self.save_config(kill_service=e.control.value))
        bar.controls.append(self.kill_service_ui)

        # tab 2
        self.paper_tab = ft.Column()

        def on_help(_):
            self.page.snack_bar = ft.SnackBar(ft.TextField(value='''
1. 输入关键词、检索的论文篇数、论文发表的最近年数
2. 开始下载，论文将保存到存档目录的paper子目录，按发表年份分类
3. 下载记录位于paper子目录下的<关键词>.json文件，重复下载将跳过已下载失败的条目，如需完全重置请删除该文件
                    ''', read_only=True, multiline=True, border=ft.InputBorder.NONE, color='white'), duration=30000)
            self.page.snack_bar.open = True
            self.page.update()

        self.paper_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.ElevatedButton('README', expand=1, on_click=on_help))

        self.paper_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        self.paper_tab.controls.append(bar := ft.Row())
        self.keyword_ui = ft.TextField(label='Keyword', value=self.config.get('keyword', 'Reality Surgery Review'),
                                       expand=1, on_change=lambda e: self.save_config(keyword=e.control.value))
        bar.controls.append(self.keyword_ui)

        self.paper_tab.controls.append(bar := ft.Row())

        _ = [ft.dropdown.Option(str(10 ** _)) for _ in range(4)]
        self.n_papers_ui = ft.Dropdown(label='Number of papers', options=_, value='10', expand=1)
        bar.controls.append(self.n_papers_ui)

        _ = [ft.dropdown.Option(str(_)) for _ in (1, 2, 3, 5, 10, 20, 30, 50, 100)]
        self.n_years_ui = ft.Dropdown(label='Number of recent years', options=_, value='10', expand=1)
        bar.controls.append(self.n_years_ui)

        self.paper_tab.controls.append(bar := ft.Row())

        def on_paper_start(_):
            if isinstance(self.paper_p, Process) and self.paper_p.is_alive():
                self.paper_p.kill()
            else:
                self.paper_in = {
                    'save': (Path(self.save_ui.value) / 'paper').as_posix(),
                    'keyword': self.keyword_ui.value,
                    'n_papers': int(self.n_papers_ui.value),
                    'n_years': int(self.n_years_ui.value),
                }
                args = (self.paper_in, self.log_q)
                self.paper_p = Process(target=paper.main, args=args, daemon=True)
                self.paper_p.start()

        self.paper_start_ui = ft.ElevatedButton(on_click=on_paper_start, expand=1)
        bar.controls.append(self.paper_start_ui)

        # tab 3
        self.embedding_tab = ft.Column()

        def on_help(_):
            self.page.snack_bar = ft.SnackBar(ft.TextField(value='''
1. 构建论文数据库，该过程将识别paper子目录下所有可识别文件，分拆并使用text-embedding-ada-002模型计算特征向量
2. 论文数据库保存在存档目录的embedding子目录下，如需完全重置请删除整个目录
3. 构建过程仅需执行一次，重复执行则跳过已录入的论文段落，避免重复消耗资源
4. 输入关键词，搜索包含相似段落的论文，按相似程度排序
                    ''', read_only=True, multiline=True, border=ft.InputBorder.NONE, color='white'), duration=30000)
            self.page.snack_bar.open = True
            self.page.update()

        self.embedding_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.ElevatedButton('README', expand=1, on_click=on_help))

        self.embedding_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        def on_embedding_build(_):
            if isinstance(self.embedding_build_p, Process) and self.embedding_build_p.is_alive():
                self.embedding_build_p.kill()
            else:
                self.embedding_build_in = {
                    'cache': (Path(self.save_ui.value) / '.cache').as_posix(),
                    'load': (Path(self.save_ui.value) / 'paper').as_posix(),
                    'embedding': (Path(self.save_ui.value) / 'embedding').as_posix(),
                }
                args = (self.embedding_build_in, self.log_q)
                self.embedding_build_p = Process(target=embedding.build, args=args, daemon=True)
                self.embedding_build_p.start()

        self.embedding_tab.controls.append(bar := ft.Row())
        self.embedding_build_ui = ft.ElevatedButton(on_click=on_embedding_build, expand=1)
        bar.controls.append(self.embedding_build_ui)

        self.embedding_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        self.embedding_tab.controls.append(bar := ft.Row())
        self.embedding_query_ui = ft.TextField(
            label='Content', value=self.config.get('query', 'Total hip'), expand=1, multiline=True,
            on_change=lambda e: self.save_config(search_input=e.control.value))
        bar.controls.append(self.embedding_query_ui)

        def on_embedding_search(_):
            if isinstance(self.embedding_search_p, Process) and self.embedding_search_p.is_alive():
                self.embedding_search_p.kill()
            else:
                self.related_papers_ui.value = ''
                self.embedding_search_in = {
                    'cache': (Path(self.save_ui.value) / '.cache').as_posix(),
                    'query': self.embedding_query_ui.value,
                    'embedding': (Path(self.save_ui.value) / 'embedding').as_posix(),
                }
                args = (self.embedding_search_in, self.log_q)
                self.embedding_search_p = Process(target=embedding.search, args=args, daemon=True)
                self.embedding_search_p.start()

        self.embedding_tab.controls.append(bar := ft.Row())
        self.embedding_search_ui = ft.ElevatedButton(on_click=on_embedding_search, expand=1)
        bar.controls.append(self.embedding_search_ui)

        self.embedding_tab.controls.append(bar := ft.Row())
        self.related_papers_ui = ft.TextField(label='Related papers', expand=1, multiline=True, read_only=True)
        bar.controls.append(self.related_papers_ui)

        self.embedding_tab.controls.append(bar := ft.Row())
        _ = [ft.dropdown.Option(str(_)) for _ in (3000, 6000, 12000)]
        self.reference_tokens_ui = ft.Dropdown(label='Reference tokens', options=_, value='3000', expand=1)
        bar.controls.append(self.reference_tokens_ui)

        def on_embedding_to_chat(_):
            text, n = embedding.text_in_tokens([_[2] for _ in self.related_texts], int(self.reference_tokens_ui.value))
            self.chat_reference_ui.value = text
            self.log_q.put(f'[EMBEDDING] reference {n} tokens')

        self.embedding_to_chat_ui = ft.ElevatedButton('Copy to chat', on_click=on_embedding_to_chat)
        bar.controls.append(self.embedding_to_chat_ui)

        # tab 4
        self.chat_tab = ft.Column()

        def on_help(_):
            self.page.snack_bar = ft.SnackBar(ft.TextField(value='''
1. 论文数据库的相关段落，可以结合提示词继续调用ChatGPT等同类服务
2. 提示词例如“请用中文重新表述：<论文段落>”
3. 注意一篇论文的token用量大约10k，超出GPT-3.5/GPT-4的限制，需分段处理
                    ''', read_only=True, multiline=True, border=ft.InputBorder.NONE, color='white'), duration=30000)
            self.page.snack_bar.open = True
            self.page.update()

        self.chat_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.ElevatedButton('README', expand=1, on_click=on_help))

        self.chat_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        self.chat_tab.controls.append(bar := ft.Row())
        self.chat_reference_ui = ft.TextField(label='Reference', expand=1, multiline=True, read_only=True, max_lines=5)
        bar.controls.append(self.chat_reference_ui)

        self.page.add(ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(
                    tab_content=ft.Text('Config'),
                    content=ft.Container(self.config_tab, padding=ft.padding.symmetric(vertical=10)),
                ),
                ft.Tab(
                    tab_content=ft.Text('Paper'),
                    content=ft.Container(self.paper_tab, padding=ft.padding.symmetric(vertical=10)),
                ),
                ft.Tab(
                    tab_content=ft.Text('Embedding'),
                    content=ft.Container(self.embedding_tab, padding=ft.padding.symmetric(vertical=10)),
                ),
                ft.Tab(
                    tab_content=ft.Text('Chat'),
                    content=ft.Container(self.chat_tab, padding=ft.padding.symmetric(vertical=10)),
                ),
            ],
            expand=1,
        ))

        # log
        self.page.controls.append(bar := ft.Row())

        bar.controls.append(ft.Text('Log output:', style=ft.TextThemeStyle.TITLE_MEDIUM))

        self.page.controls.append(bar := ft.Row())

        self.log_ui = ft.ListView(expand=1, auto_scroll=True, on_scroll_interval=1, height=100)
        bar.controls.append(self.log_ui)

        self.page.update()
        self.loop()

    def loop(self):
        if isinstance(self.paper_p, Process) and self.paper_p.is_alive():
            self.paper_start_ui.text = 'Cancel'
        else:
            self.paper_start_ui.text = 'Download papers'

        if isinstance(self.embedding_build_p, Process) and self.embedding_build_p.is_alive():
            self.embedding_build_ui.text = 'Cancel'
        else:
            self.embedding_build_ui.text = 'Build paper database'

        if isinstance(self.embedding_search_p, Process) and self.embedding_search_p.is_alive():
            self.embedding_search_ui.text = 'Cancel'
        else:
            self.embedding_search_ui.text = 'Search related papers'

        try:
            log: str = self.log_q.get(block=False)
            if isinstance(log, dict):
                if _ := log.get('related papers', None):
                    self.related_texts = _
                    self.related_papers_ui.value = '\n'.join({f'{_[0]} {_[1]}' for _ in self.related_texts})
            else:
                self.log_ui.controls.append(ft.Text(log))

                if log.startswith('ERROR'):
                    self.page.dialog.content = ft.Text(log)
                    self.page.dialog.open = True
        except Empty:
            pass

        self.page.update()
        threading.Timer(0.1, self.loop).start()


def main():
    freeze_support()

    def about_to_exit():
        if app.config.get('kill_service', False):
            default_server.stop()
            for proc in psutil.process_iter():
                if 'milvus' in proc.name().lower():
                    proc.kill()

    atexit.register(about_to_exit)

    app = App()
    ft.app(target=app)
