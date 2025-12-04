import os
import json
import urllib.parse
import re
from datetime import datetime
from pathlib import Path

PDF_DIR = "papers"
METADATA_FILE = "metadata.json"
OUTPUT_HTML = "index.html"
TEMPLATE_HTML = "template.html"

os.makedirs(PDF_DIR, exist_ok=True)

# 根据文件名自动推断会议/年份（如果匹配不到则返回 None）
def infer_venue_and_year(fname):
    fname_lower = fname.lower()
    patterns = [
        (r"ase(?:20)?(\d{2})", "ASE"),
        (r"icse(?:20)?(\d{2})", "ICSE"),
        (r"issta(?:20)?(\d{2})", "ISSTA"),
        (r"acl(?:20)?(\d{2})", "ACL"),
        (r"arxiv(?:20)?(\d{2})", "arXiv"),
        (r"nip(?:s)?(?:20)?(\d{2})", "NeurIPS"),
        (r"cvpr(?:20)?(\d{2})", "CVPR"),
        (r"iccv(?:20)?(\d{2})", "ICCV"),
        (r"eccv(?:20)?(\d{2})", "ECCV"),
        (r"aaai(?:20)?(\d{2})", "AAAI"),
        (r"icra(?:20)?(\d{2})", "ICRA"),
        (r"siggraph(?:20)?(\d{2})", "SIGGRAPH"),
    ]
    for pat, venue_name in patterns:
        match = re.search(pat, fname_lower)
        if match:
            year_suffix = match.group(1)
            year = f"20{year_suffix}"
            return venue_name, year
    return None, None

# 加载或创建 metadata
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {}

# 扫描 PDF 文件
papers = []
pdf_files = []
for root, _, files in os.walk(PDF_DIR):
    for fname in files:
        if fname.lower().endswith(".pdf"):
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, PDF_DIR)
            rel_path = rel_path.replace(os.sep, "/")
            pdf_files.append((rel_path, fname))

pdf_files.sort(key=lambda item: item[0].lower())
legacy_keys_to_remove = set()

for rel_path, fname in pdf_files:
    key = rel_path.lower()
    legacy_key = fname.lower()

    info = metadata.get(key) or metadata.get(legacy_key, {})
    if legacy_key in metadata and key != legacy_key:
        legacy_keys_to_remove.add(legacy_key)

    # URL 编码路径
    quoted_rel_path = "/".join(urllib.parse.quote(part) for part in rel_path.split("/"))

    # 自动生成 tag：如果 metadata 中没有 tags 或为空，则使用 PDF 所在的一级文件夹
    folder_tag = rel_path.split("/")[0] if "/" in rel_path else "unsorted"
    tags = info.get("tags")
    if not tags:
        tags = [folder_tag]

    # 自动从文件名推断年份和会议
    year = info.get("year")
    venue = info.get("venue")
    if not year or not venue:
        inferred_venue, inferred_year = infer_venue_and_year(fname)
        if not year:
            year = inferred_year  # 只有匹配到才填
        if not venue:
            venue = inferred_venue  # 只有匹配到才填

    paper = {
        "file_key": key,
        "title": info.get("title", os.path.splitext(fname)[0]),
        "authors": info.get("authors", "Unknown"),
        "year": year if year else "",      # 未匹配到保持空
        "venue": venue if venue else "",   # 未匹配到保持空
        "tags": tags,
        "pdf": f"{PDF_DIR}/{quoted_rel_path}",
        "pdf_local": f"{PDF_DIR}/{quoted_rel_path}",
        "read": info.get("read", False),
        "bib": info.get("bib", ""),
        "notes": info.get("notes", "")
    }

    papers.append(paper)
    metadata[key] = paper

for legacy_key in legacy_keys_to_remove:
    metadata.pop(legacy_key, None)

# 保存 metadata.json
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# 渲染 HTML
if not os.path.exists(TEMPLATE_HTML):
    raise FileNotFoundError(f"{TEMPLATE_HTML} 不存在")

with open(TEMPLATE_HTML, "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("const EMBEDDED_PAPERS = [];", f"const EMBEDDED_PAPERS = {json.dumps(papers, ensure_ascii=False)};")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ 成功生成 {OUTPUT_HTML}，共 {len(papers)} 篇论文")
