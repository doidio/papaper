import json
import os
import sys
import threading
from multiprocessing import Queue, Process
from pathlib import Path
from queue import Empty

import flet as ft

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

        self.config_path = Path(sys.executable).parent.parent / 'config.json'

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

        self.page.title = 'Papaper 1.2.0'
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
3. 设置Save路径到你的存档目录，其中将会保存下载的论文和用于相似性搜索的数据库
                    ''', read_only=True, multiline=True, border=ft.InputBorder.NONE, color='white'), duration=30000)
            self.page.snack_bar.open = True
            self.page.update()

        self.config_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.ElevatedButton('README', expand=1, on_click=on_help))

        self.config_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        self.config_tab.controls.append(bar := ft.Row())

        _ = Path(sys.executable).parent.parent / 'save'
        _ = _.absolute().as_posix()
        self.save_ui = ft.TextField(label='Save', value=self.config.get('save', _), expand=1,
                                    on_change=lambda e: self.save_config(save=e.control.value))
        bar.controls.append(self.save_ui)

        def on_open_save(_):
            os.startfile(self.save_ui.value)

        bar.controls.append(ft.ElevatedButton('Open', on_click=on_open_save))

        # tab 2
        self.paper_tab = ft.Column()

        def on_help(_):
            self.page.snack_bar = ft.SnackBar(ft.TextField(value='''
1. 输入关键词、检索的论文篇数、论文发表的最近年数
2. 开始下载，论文将保存到存档目录的documents子目录，按发表年份分类
3. 下载记录位于documents子目录下的<关键词>.json文件，重复下载将跳过已下载失败的条目，如需完全重置请删除该文件
4. 不会消耗付费资源
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
                    'save': (Path(self.save_ui.value) / 'documents').as_posix(),
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
1. 构建数据库，该过程将识别documents子目录下所有可识别的文档，文档越多构建耗时越长
2. 数据库保存在embedding子目录下，每次构建都将覆盖旧文件
3. 输入查询文本，在数据库中搜索相似的文本段落，按相似度排序
4. 根据chat模型的token限制，例如GPT3-5 16k，选择合适的资源token数量，一键复制作为引用资料
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
                    'load': (Path(self.save_ui.value) / 'documents').as_posix(),
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
            label='Query', value=self.config.get('query', 'Total hip'), expand=1, multiline=True,
            on_change=lambda e: self.save_config(search_input=e.control.value))
        bar.controls.append(self.embedding_query_ui)

        def on_embedding_search(_):
            if isinstance(self.embedding_search_p, Process) and self.embedding_search_p.is_alive():
                self.embedding_search_p.kill()
            else:
                self.related_documents_ui.value = ''
                self.embedding_search_in = {
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
        self.related_documents_ui = ft.TextField(label='Related documents', expand=1, multiline=True, read_only=True)
        bar.controls.append(self.related_documents_ui)

        self.embedding_tab.controls.append(bar := ft.Row())
        _ = [ft.dropdown.Option(str(_)) for _ in (2500, 5000, 10000, 20000)]
        self.reference_tokens_ui = ft.Dropdown(label='Resource tokens', options=_, value='2500', expand=1)
        bar.controls.append(self.reference_tokens_ui)

        def on_embedding_to_chat(_):
            text, n = embedding.text_in_tokens([_[2] for _ in self.related_texts], int(self.reference_tokens_ui.value))
            self.chat_resource_ui.value = text
            self.log_q.put(f'[EMBEDDING] reference {n} tokens')

        self.embedding_to_chat_ui = ft.ElevatedButton('Copy to chat', on_click=on_embedding_to_chat)
        bar.controls.append(self.embedding_to_chat_ui)

        # tab 4
        self.chat_tab = ft.Column()

        def on_help(_):
            self.page.snack_bar = ft.SnackBar(ft.TextField(value='''
1. 修改提示词和问题，查看完整对话，复制到任何大语言聊天应用中使用
                    ''', read_only=True, multiline=True, border=ft.InputBorder.NONE, color='white'), duration=30000)
            self.page.snack_bar.open = True
            self.page.update()

        self.chat_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.ElevatedButton('README', expand=1, on_click=on_help))

        self.chat_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Divider())

        self.chat_tab.controls.append(bar := ft.Row())
        self.chat_question_prompt_ui = ft.TextField(
            label='Question prompt', expand=1, multiline=True, max_lines=3,
            value=self.config.get('question_prompt', '你是一名学者，请用中文为我提供帮助。'),
            on_change=lambda e: self.save_config(question_prompt=e.control.value))
        bar.controls.append(self.chat_question_prompt_ui)

        self.chat_tab.controls.append(bar := ft.Row())
        self.chat_question_ui = ft.TextField(
            label='Question', expand=1, multiline=True, max_lines=3,
            value=self.config.get('question', '根据资料重新组织一段论述。'),
            on_change=lambda e: self.save_config(question=e.control.value))
        bar.controls.append(self.chat_question_ui)

        self.chat_tab.controls.append(bar := ft.Row())
        self.chat_resource_prompt_ui = ft.TextField(
            label='Resource prompt',
            value=self.config.get('resource_prompt', '请充分阅读理解资料，在你的回答中不能体现你事先阅读了这些资料。'
                                                     '资料如下：'),
            expand=1, multiline=True, max_lines=3,
            on_change=lambda e: self.save_config(resource_prompt=e.control.value))
        bar.controls.append(self.chat_resource_prompt_ui)

        self.chat_tab.controls.append(bar := ft.Row())
        self.chat_resource_ui = ft.TextField(
            label='Resource', expand=1, multiline=True, read_only=True, max_lines=3,
            value=self.config.get('resource', ''),
            on_change=lambda e: self.save_config(resource=e.control.value))
        bar.controls.append(self.chat_resource_ui)

        def on_chat_clipboard(_):
            text = '\n'.join([self.chat_question_prompt_ui.value,
                              self.chat_question_ui.value,
                              self.chat_resource_prompt_ui.value,
                              self.chat_resource_ui.value])

            self.page.dialog.content = ft.TextField(value=text, read_only=True, multiline=True, expand=1)
            self.page.dialog.open = True
            self.page.update()

        self.chat_tab.controls.append(bar := ft.Row())
        self.chat_clipboard_ui = ft.ElevatedButton('View dialog', on_click=on_chat_clipboard)
        bar.controls.append(self.chat_clipboard_ui)

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
            self.embedding_build_ui.text = 'Build database'

        if isinstance(self.embedding_search_p, Process) and self.embedding_search_p.is_alive():
            self.embedding_search_ui.text = 'Cancel'
        else:
            self.embedding_search_ui.text = 'Search related documents'

        try:
            log: str = self.log_q.get(block=False)
            if isinstance(log, dict):
                if _ := log.get('related documents', None):
                    self.related_texts = _
                    self.related_documents_ui.value = '\n'.join({f'{_[0]} {_[1]}' for _ in self.related_texts})
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
    app = App()
    ft.app(target=app)
