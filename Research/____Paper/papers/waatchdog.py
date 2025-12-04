import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

PDF_DIR = "papers"  # 监控目录
BUILD_SCRIPT = "build.py"  # 构建脚本

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            print(f"📄 检测到新增 PDF: {event.src_path}")
            self.run_build()

    def on_moved(self, event):
        # 支持移动文件到目录
        if not event.is_directory and event.dest_path.lower().endswith(".pdf"):
            print(f"📄 检测到新增 PDF（移动文件）: {event.dest_path}")
            self.run_build()

    def run_build(self):
        print("⚡ 自动执行 build.py ...")
        subprocess.run(["python", BUILD_SCRIPT])

if __name__ == "__main__":
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, PDF_DIR, recursive=True)
    observer.start()
    print(f"🔔 开始监控 {PDF_DIR} 下的 PDF 文件变化 ...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
