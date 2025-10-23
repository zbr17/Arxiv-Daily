import feedparser
import pytz
from datetime import datetime
import os
import re
import logging
import sys
import math
from volcenginesdkarkruntime import Ark

# --- 配置项 ---

# ArXiv 分类
CATEGORIES_TO_FETCH = ["cs.CV", "cs.LG", "cs.AI", "cs.RO"]
# RSS 过滤器，'Announce Type: new' 表示只看新提交的论文
RSS_FILTER_KEY = "Announce Type: new"
# LLM 模型 ID
MODEL_ID = "doubao-seed-1-6-thinking-250715"
# Prompt 模板文件
PROMPT_FILE = "prompt.txt"
# 每批次处理的论文档案数（防止单次API调用过长）
BATCH_SIZE = 50

# --- 全局变量 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_api_key():
    """从环境变量中获取 API 密钥，增强安全性。"""
    api_key = os.environ.get('VOLC_API_KEY')
    if not api_key:
        logging.error("环境变量 VOLC_API_KEY 未设置。")
        sys.exit(1)
    return api_key

def get_prompt_template():
    """加载 Prompt 模板文件。"""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt 文件 {PROMPT_FILE} 未找到。")
        sys.exit(1)
    except Exception as e:
        logging.error(f"读取 Prompt 文件时出错: {e}")
        sys.exit(1)

def get_recent_papers(category, filter_key):
    """从 ArXiv RSS feed 中获取指定分类的最新论文列表。"""
    base_url = 'http://export.arxiv.org/rss/'
    try:
        feed = feedparser.parse(base_url + category)
        recent_papers = []
        for entry in feed.entries:
            # 根据摘要中的关键词过滤（例如 'new', 'cross-list')
            if filter_key in entry.summary:
                published_time = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
                paper = {
                    'title': entry.title.replace('\n', ' ').strip(),
                    'authors': ', '.join(author.name for author in entry.authors),
                    'published': published_time.strftime('%Y-%m-%d'),
                    'summary': entry.summary.replace('\n', ' ').strip(),
                    'link': entry.link
                }
                recent_papers.append(paper)
        logging.info(f"从 {category} 成功获取 {len(recent_papers)} 篇新论文。")
        return recent_papers
    except Exception as e:
        logging.error(f"获取 ArXiv (分类: {category}) RSS时出错: {e}")
        return []

def list_to_markdown_for_llm(papers):
    """将论文列表转换为供给 LLM 阅读的 Markdown 格式。"""
    markdown = ""
    for i, paper in enumerate(papers, 1):
        markdown += f"### 论文 {i}/{len(papers)}\n"
        markdown += f"- 标题: {paper['title']}\n"
        markdown += f"- 作者: {paper['authors']}\n"
        markdown += f"- 时间：{paper['published']}\n"
        markdown += f"- 链接: [{paper['link']}]({paper['link']})\n"
        markdown += f"- 摘要: {paper['summary']}\n\n"
    return markdown

def get_llm_response(query, api_key):
    """调用火山方舟大模型 API。"""
    try:
        client = Ark(api_key=api_key)
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        total_tokens = completion.usage.total_tokens
        content = completion.choices[0].message.content
        return content, total_tokens
    except Exception as e:
        logging.error(f"调用大模型 API 时出错: {e}")
        return f"[ERROR: LLM API FAILED {e}]", 0

def parse_llm_output(output_text):
    """
    使用正则表达式解析 LLM 的结构化输出。
    满足要求 (5)：使用特殊分隔符，容错率高。
    """
    papers = []
    # 使用 re.DOTALL 标志使 . 能够匹配换行符
    paper_blocks = re.findall(r"\[PAPER_START\](.*?)\[PAPER_END\]", output_text, re.DOTALL)
    
    for block in paper_blocks:
        paper_data = {}
        try:
            title = re.search(r"Title:\s*(.*)", block, re.IGNORECASE)
            authors = re.search(r"Authors:\s*(.*)", block, re.IGNORECASE)
            published = re.search(r"Published:\s*(.*)", block, re.IGNORECASE)
            link = re.search(r"Link:\s*(.*)", block, re.IGNORECASE)
            reason = re.search(r"Reason:\s*(.*)", block, re.DOTALL | re.IGNORECASE)
            score = re.search(r"Score:\s*(\d+\.?\d*)", block, re.IGNORECASE)
            field = re.search(r"Field:\s*(.*)", block, re.IGNORECASE)

            if all([title, authors, published, link, reason, score, field]):
                paper_data['title'] = title.group(1).strip()
                paper_data['authors'] = authors.group(1).strip()
                paper_data['published'] = published.group(1).strip()
                paper_data['link'] = link.group(1).strip()
                paper_data['reason'] = reason.group(1).strip()
                paper_data['score'] = float(score.group(1))
                paper_data['field'] = field.group(1).strip()
                papers.append(paper_data)
            else:
                logging.warning(f"解析论文块失败，缺少字段: {block}")
        except Exception as e:
            logging.warning(f"解析论文块时出现异常: {e}\n块内容: {block}")
            
    return papers

def group_and_sort_papers(papers):
    """
    对解析后的论文列表进行排序和分组。
    满足要求 (2)：按研究方向分类。
    """
    # 1. 按得分排序（全局排序）
    papers.sort(key=lambda x: x['score'], reverse=True)
    
    # 2. 按领域分组
    grouped_papers = {}
    for paper in papers:
        field = paper.get('field', 'Other')
        if field not in grouped_papers:
            grouped_papers[field] = []
        grouped_papers[field].append(paper)
        
    return grouped_papers

def generate_readme(grouped_papers, total_papers_read, total_tokens, update_time):
    """生成 README.md 文件的内容。"""
    
    md_content = f"# ArXiv 每日推荐\n\n"
    md_content += f"> 更新于北京时间：{update_time}\n"
    md_content += f"> 已自动阅读了 {total_papers_read} 篇最新的论文。\n"
    md_content += f"> 使用模型：{MODEL_ID} | 消耗 Tokens：{total_tokens}\n\n"
    
    if not grouped_papers:
        md_content += "今日未发现相关论文。\n"
        return md_content

    # 添加导航（可选，但对 README 友好）
    md_content += "## 快速导航\n\n"
    for field in grouped_papers:
        # 创建一个简单的锚点
        anchor = re.sub(r'\s+', '-', field).lower()
        md_content += f"- [{field}](#{anchor})\n"
    md_content += "\n"

    # 生成论文列表
    for field, papers in grouped_papers.items():
        anchor = re.sub(r'\s+', '-', field).lower()
        md_content += f"<h2 id='{anchor}'>{field}</h2>\n\n"
        for paper in papers:
            md_content += f"### [Score: {paper['score']}/10] {paper['title']}\n"
            md_content += f"- **Authors:** {paper['authors']}\n"
            md_content += f"- **Published:** {paper['published']}\n"
            md_content += f"- **Link:** [{paper['link']}]({paper['link']})\n"
            md_content += f"- **Reason:** {paper['reason']}\n\n"
            
    return md_content

def generate_html(grouped_papers, total_papers_read, total_tokens, update_time):
    """
    生成 index.html 文件的内容 (GitHub Page)。
    满足要求 (2)：生成网页，带导航栏和美化。
    """
    
    # CSS 样式（内联）
    css_style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; background-color: #f6f8fa; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 20px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        h1, h2, h3 { border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }
        h1 { font-size: 2em; }
        h2 { font-size: 1.5em; margin-top: 2.5em; }
        h3 { font-size: 1.2em; border-bottom: 1px solid #eee; }
        .navbar { background-color: #333; overflow: hidden; position: sticky; top: 0; z-index: 100; }
        .navbar a { float: left; display: block; color: white; text-align: center; padding: 14px 16px; text-decoration: none; font-size: 17px; }
        .navbar a:hover { background-color: #ddd; color: black; }
        .navbar a.active { background-color: #007bff; color: white; }
        .meta-info { font-size: 0.9em; color: #586069; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
        .paper-card { margin-bottom: 1.5em; padding-bottom: 1em; }
        .paper-card p { margin: 0.5em 0; }
        .paper-card a { color: #007bff; text-decoration: none; font-weight: bold; }
        .paper-card a:hover { text-decoration: underline; }
        .score { font-weight: bold; color: #d9534f; }
        .field-section { padding-top: 60px; margin-top: -60px; } /* 确保锚点跳转时不会被导航栏遮挡 */
    </style>
    """

    # HTML 头部
    html_content = f"<!DOCTYPE html>\n<html lang='zh-CN'>\n<head>\n"
    html_content += f"<meta charset='UTF-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    html_content += f"<title>ArXiv 每日推荐</title>\n{css_style}\n</head>\n<body>\n"

    # 导航栏
    html_content += "<div class='navbar'>\n"
    if grouped_papers:
        for i, field in enumerate(grouped_papers):
            safe_id = re.sub(r'[^a-zA-Z0-9\s-]', '', field).replace(' ', '-')
            active_class = "class='active'" if i == 0 else ""
            html_content += f"<a href='#{safe_id}' {active_class}>{field}</a>\n"
    html_content += "</div>\n"

    # 页面主体
    html_content += "<div class='container'>\n"
    html_content += "<h1>ArXiv 每日推荐</h1>\n"
    html_content += f"<div class='meta-info'>"
    html_content += f"<p>更新于北京时间：{update_time}</p>\n"
    html_content += f"<p>已自动阅读了 {total_papers_read} 篇最新的论文。</p>\n"
    html_content += f"<p>使用模型：{MODEL_ID} | 消耗 Tokens：{total_tokens}</p>\n"
    html_content += "</div>\n"

    if not grouped_papers:
        html_content += "<h2>今日未发现相关论文。</h2>\n"
    else:
        for field, papers in grouped_papers.items():
            safe_id = re.sub(r'[^a-zA-Z0-9\s-]', '', field).replace(' ', '-')
            html_content += f"<div id='{safe_id}' class='field-section'>\n<h2>{field}</h2>\n"
            for paper in papers:
                html_content += "<div class='paper-card'>\n"
                html_content += f"<h3><span class='score'>[Score: {paper['score']}/10]</span> {paper['title']}</h3>\n"
                html_content += f"<p><strong>Authors:</strong> {paper['authors']}</p>\n"
                html_content += f"<p><strong>Published:</strong> {paper['published']}</p>\n"
                html_content += f"<p><strong>Reason:</strong> {paper['reason']}</p>\n"
                html_content += f"<p><a href='{paper['link']}' target='_blank'>阅读论文 &raquo;</a></p>\n"
                html_content += "</div>\n"
            html_content += "</div>\n"

    html_content += "</div>\n" # end .container
    
    # JavaScript (用于导航栏高亮)
    html_content += """
    <script>
    document.addEventListener('DOMContentLoaded', (event) => {
        const sections = document.querySelectorAll('.field-section');
        const navLinks = document.querySelectorAll('.navbar a');

        function changeLinkState() {
            let index = sections.length;

            while(--index && window.scrollY + 100 < sections[index].offsetTop) {} // 100 as offset

            navLinks.forEach((link) => link.classList.remove('active'));
            // Check if navLinks[index] exists
            if (navLinks[index]) {
                navLinks[index].classList.add('active');
            }
        }

        changeLinkState();
        window.addEventListener('scroll', changeLinkState);
    });
    </script>
    """
    html_content += "</body>\n</html>"
    return html_content

def write_file(filename, content):
    """安全的写入文件。"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"成功写入文件: {filename}")
    except IOError as e:
        logging.error(f"写入文件 {filename} 时出错: {e}")

def main():
    logging.info("--- 开始执行每日 ArXiv 推荐任务 ---")
    
    # 0. 加载配置
    api_key = get_api_key()
    prompt_template = get_prompt_template()
    
    # 1. 获取所有分类的论文
    all_papers = []
    for category in CATEGORIES_TO_FETCH:
        all_papers.extend(get_recent_papers(category, RSS_FILTER_KEY))
    
    # 去重
    seen_links = set()
    unique_papers = []
    for paper in all_papers:
        if paper['link'] not in seen_links:
            unique_papers.append(paper)
            seen_links.add(paper['link'])
    
    total_papers_read = len(unique_papers)
    logging.info(f"共获取到 {total_papers_read} 篇不重复的新论文。")
    
    if total_papers_read == 0:
        logging.info("未获取到新论文，任务提前结束。")
        update_time_str = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        grouped_papers = {}
        readme_content = generate_readme(grouped_papers, 0, 0, update_time_str)
        html_content = generate_html(grouped_papers, 0, 0, update_time_str)
        write_file("README.md", readme_content)
        write_file("index.html", html_content)
        write_file("RAW.md", "# 今日无新论文\n") # 清空 RAW.md
        return

    # 2. 分批处理（满足要求 1）
    all_llm_outputs = []
    total_tokens_used = 0
    raw_md_content = "" # 用于保存发给 LLM 的原始输入
    
    num_batches = math.ceil(total_papers_read / BATCH_SIZE)
    logging.info(f"开始分批处理，共 {num_batches} 批，每批 {BATCH_SIZE} 篇。")
    
    for i in range(num_batches):
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE
        batch_papers = unique_papers[batch_start:batch_end]
        
        logging.info(f"正在处理第 {i+1}/{num_batches} 批...")
        
        # 准备输入
        batch_llm_input_md = list_to_markdown_for_llm(batch_papers)
        raw_md_content += f"# --- 批次 {i+1}/{num_batches} ---\n\n{batch_llm_input_md}\n\n"
        
        query = prompt_template.format(
            batch_num=i + 1,
            total_batches=num_batches,
            paper_list=batch_llm_input_md
        )
        
        # 调用 LLM
        response_content, tokens = get_llm_response(query, api_key)
        total_tokens_used += tokens
        
        if "[ERROR:" not in response_content and "[NO_RELEVANT_PAPERS]" not in response_content:
            all_llm_outputs.append(response_content)
            
    logging.info(f"所有批次处理完毕。总消耗 Tokens: {total_tokens_used}")
    
    # 写入原始输入（用于调试）
    write_file("RAW.md", raw_md_content)

    # 3. 解析和后处理
    full_llm_output = "\n".join(all_llm_outputs)
    parsed_papers = parse_llm_output(full_llm_output)
    logging.info(f"从 LLM 输出中成功解析出 {len(parsed_papers)} 篇相关论文。")
    
    grouped_and_sorted_papers = group_and_sort_papers(parsed_papers)
    
    # 4. 生成和保存文件（满足要求 2）
    update_time_str = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    
    # 生成 README.md
    readme_content = generate_readme(grouped_and_sorted_papers, total_papers_read, total_tokens_used, update_time_str)
    write_file("README.md", readme_content)
    
    # 生成 index.html
    html_content = generate_html(grouped_and_sorted_papers, total_papers_read, total_tokens_used, update_time_str)
    write_file("index.html", html_content)
    
    logging.info("--- 任务执行完毕 ---")

if __name__ == "__main__":
    main()