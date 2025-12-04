import os
import json
import shutil

PDF_DIR = "papers"
METADATA_FILE = "metadata.json"

def main():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("metadata.json 不存在，请先运行主脚本生成 metadata。")

    # 读取 metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    moved_count = 0
    created_folders = set()

    for key, info in metadata.items():
        # metadata 的 key 是相对路径 "xxx/yyy.pdf"
        pdf_rel_path = key
        pdf_abs_path = os.path.join(PDF_DIR, pdf_rel_path)

        if not os.path.exists(pdf_abs_path):
            print(f"⚠️ 找不到文件 {pdf_abs_path}，跳过")
            continue

        tags = info.get("tags", ["unsorted"])
        if not tags:
            target_tag = "unsorted"
        else:
            target_tag = tags[0]  # 取第一 tag

        # 目标文件夹
        target_folder = os.path.join(PDF_DIR, target_tag)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder, exist_ok=True)
            created_folders.add(target_tag)

        # 目标路径
        file_name = os.path.basename(pdf_rel_path)
        target_path = os.path.join(target_folder, file_name)

        # 避免重复移动或覆盖
        if os.path.abspath(pdf_abs_path) == os.path.abspath(target_path):
            continue  # 文件已经在正确位置

        # 如果目标文件存在，避免覆盖（你可以改 rename）
        if os.path.exists(target_path):
            print(f"⚠️ 目标已存在，跳过移动: {target_path}")
            continue

        # 移动
        shutil.move(pdf_abs_path, target_path)
        moved_count += 1
        print(f"📁 Moved: {pdf_rel_path} → {target_tag}/{file_name}")

    print("\n======================")
    print("分类完成")
    print(f"📦 新建文件夹: {created_folders}")
    print(f"📚 处理文件数: {moved_count}")
    print("======================")

if __name__ == "__main__":
    main()
