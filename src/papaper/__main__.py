import os
import threading
from multiprocessing import Queue, Process, freeze_support
from pathlib import Path
from queue import Empty

import flet as ft

from papaper import paper


class App:
    def __init__(self):
        self.page: ft.Page

        self.paper_p = None
        self.paper_in = None
        self.paper_log = Queue()

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

        self.page.controls.append(bar := ft.Row())

        _ = Path(os.getcwd()) / 'save'
        _ = _.absolute().as_posix()
        self.save_in = ft.TextField(label='Save in', value=_, expand=1)
        bar.controls.append(self.save_in)

        # tab 0
        self.paper_tab = ft.Column()
        self.paper_tab.controls.append(bar := ft.Row())

        self.keyword_ui = ft.TextField(label='Keyword', value='Generative Pre-trained Transformer', expand=1)
        bar.controls.append(self.keyword_ui)

        self.paper_tab.controls.append(bar := ft.Row())

        _ = [ft.dropdown.Option(str(10 ** _)) for _ in range(4)]
        self.n_papers_ui = ft.Dropdown(label='Number of papers', options=_, value='10', expand=1)
        bar.controls.append(self.n_papers_ui)

        _ = [ft.dropdown.Option(str(_)) for _ in (1, 2, 3, 5, 10, 20, 30, 50, 100)]
        self.n_years_ui = ft.Dropdown(label='Number of recent years', options=_, value='10', expand=1)
        bar.controls.append(self.n_years_ui)

        self.paper_tab.controls.append(bar := ft.Row())

        def on_start_or_cancel(_):
            if isinstance(self.paper_p, Process) and self.paper_p.is_alive():
                self.paper_p.kill()
            else:
                self.log_ui.controls.clear()
                self.paper_in = {
                    'save_in': (Path(self.save_in.value) / 'papers').as_posix(),
                    'keyword': self.keyword_ui.value,
                    'n_papers': int(self.n_papers_ui.value),
                    'n_years': int(self.n_years_ui.value),
                }
                self.paper_p = Process(target=paper.main, args=(self.paper_in, self.paper_log), daemon=True)
                self.paper_p.start()

        self.start_ui = ft.ElevatedButton('Start', on_click=on_start_or_cancel, expand=1)
        bar.controls.append(self.start_ui)

        self.paper_tab.controls.append(bar := ft.Row())
        bar.controls.append(ft.Text('Log output:', style=ft.TextThemeStyle.TITLE_MEDIUM))

        self.progress_ui = ft.ProgressRing(visible=False)
        bar.controls.append(self.progress_ui)

        self.log_ui = ft.ListView(expand=1, auto_scroll=True)
        self.paper_tab.controls.append(self.log_ui)

        # tab 1
        self.embedding_tab = ft.Column()
        self.embedding_tab.controls.append(bar := ft.Row())

        self.page.add(ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(
                    tab_content=ft.Text('Paper'),
                    content=ft.Container(self.paper_tab, padding=ft.padding.symmetric(vertical=10)),
                ),
                ft.Tab(
                    tab_content=ft.Text('Embedding'),
                    content=ft.Container(self.embedding_tab, padding=ft.padding.symmetric(vertical=10)),
                ),
            ],
            expand=1,
        ))

        self.page.update()
        self.loop()

    def loop(self):
        if isinstance(self.paper_p, Process) and self.paper_p.is_alive():
            self.start_ui.text = 'Cancel'
            self.progress_ui.visible = True
        else:
            self.start_ui.text = 'Start'
            self.progress_ui.visible = False

        try:
            log: str = self.paper_log.get(block=False)
            self.log_ui.controls.append(ft.Text(log, style=ft.TextThemeStyle.BODY_MEDIUM))

            if log.startswith('ERROR'):
                self.page.dialog.content = ft.Text(log)
                self.page.dialog.open = True
        except Empty:
            pass

        self.page.update()
        threading.Timer(0.1, self.loop).start()


if __name__ == '__main__':
    freeze_support()
    app = App()
    ft.app(target=app)
