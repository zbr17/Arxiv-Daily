import feedparser
import pytz
from datetime import datetime, timedelta
import os
import re
import logging
import sys
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from volcenginesdkarkruntime import Ark

# --- 配置项 ---
CATEGORIES_TO_FETCH = ["cs.CV", "cs.LG", "cs.AI", "cs.RO"]
RSS_FILTER_KEY = "Announce Type: new"
MODEL_ID = "doubao-seed-1-6-thinking-250715"
PROMPT_FILE = "prompt.txt"
BATCH_SIZE = 50
HISTORY_DIR = "./history"
MAX_HISTORY_DAYS = 50

# 预定义分类列表
PREDEFINED_CATEGORIES = [
    "深度学习理论",
    "深度学习可解释性", 
    "自动驾驶与大模型",
    "多模态大模型"
]

# --- 全局变量 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArXivProcessor:
    def __init__(self):
        self.api_key = self.get_api_key()
        self.prompt_template = self.get_prompt_template()
        self.vectorizer = TfidfVectorizer()
        self._fit_vectorizer()
        
    def _fit_vectorizer(self):
        """训练TF-IDF向量化器"""
        # 为每个类别添加一些可能的变体，以提高匹配准确性
        category_texts = PREDEFINED_CATEGORIES + [
            "深度学习", "神经网络理论", "优化算法", "网络架构",
            "可解释性", "解释性AI", "SHAP", "白盒解释",
            "自动驾驶", "无人驾驶", "车载大模型", "多模态自动驾驶",
            "多模态", "视觉语言模型", "图像生成", "GUI智能体"
        ]
        self.vectorizer.fit(category_texts)
        
    def get_api_key(self):
        """从环境变量中获取 API 密钥"""
        api_key = os.environ.get('VOLC_API_KEY')
        if not api_key:
            logging.error("环境变量 VOLC_API_KEY 未设置。")
            sys.exit(1)
        return api_key

    def get_prompt_template(self):
        """加载 Prompt 模板文件"""
        try:
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt 文件 {PROMPT_FILE} 未找到。")
            sys.exit(1)
        except Exception as e:
            logging.error(f"读取 Prompt 文件时出错: {e}")
            sys.exit(1)

    def get_recent_papers(self, category, filter_key):
        """从 ArXiv RSS feed 中获取指定分类的最新论文列表"""
        base_url = 'http://export.arxiv.org/rss/'
        try:
            feed = feedparser.parse(base_url + category)
            recent_papers = []
            for entry in feed.entries:
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

    def list_to_markdown_for_llm(self, papers):
        """将论文列表转换为供给 LLM 阅读的 Markdown 格式"""
        markdown = ""
        for i, paper in enumerate(papers, 1):
            markdown += f"### 论文 {i}/{len(papers)}\n"
            markdown += f"- 标题: {paper['title']}\n"
            markdown += f"- 作者: {paper['authors']}\n"
            markdown += f"- 时间：{paper['published']}\n"
            markdown += f"- 链接: [{paper['link']}]({paper['link']})\n"
            markdown += f"- 摘要: {paper['summary']}\n\n"
        return markdown

    def list_to_raw_markdown(self, papers):
        """将论文列表转换为原始 RAW.md 格式"""
        markdown = "# 今日原始论文列表\n\n"
        markdown += f"共获取到 {len(papers)} 篇新论文\n\n"
        
        for i, paper in enumerate(papers, 1):
            markdown += f"## 论文 {i}\n"
            markdown += f"- **标题**: {paper['title']}\n"
            markdown += f"- **作者**: {paper['authors']}\n"
            markdown += f"- **发布时间**: {paper['published']}\n"
            markdown += f"- **链接**: {paper['link']}\n"
            markdown += f"- **摘要**: {paper['summary']}\n\n"
        return markdown

    def get_llm_response(self, query):
        """调用火山方舟大模型 API"""
        try:
            client = Ark(api_key=self.api_key)
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

    def normalize_field(self, field):
        """使用embedding距离规范化研究方向分类"""
        if not field or not field.strip():
            return "其他"
        
        field = field.strip()
        
        # 如果已经是预定义分类，直接返回
        if field in PREDEFINED_CATEGORIES:
            return field
        
        # 检查是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in field)
        if not has_chinese:
            logging.warning(f"分类 '{field}' 不包含中文字符，尝试映射到最接近的中文分类")
        
        try:
            # 计算输入字段与预定义分类的embedding相似度
            input_vec = self.vectorizer.transform([field])
            category_vecs = self.vectorizer.transform(PREDEFINED_CATEGORIES)
            
            # 计算余弦相似度
            similarities = cosine_similarity(input_vec, category_vecs)[0]
            
            # 找到最相似的分类
            best_match_idx = np.argmax(similarities)
            best_match = PREDEFINED_CATEGORIES[best_match_idx]
            max_similarity = similarities[best_match_idx]
            
            # 设置相似度阈值
            threshold = 0.3  # 可以根据需要调整
            
            if max_similarity >= threshold:
                logging.info(f"分类 '{field}' 被规范化为 '{best_match}' (相似度: {max_similarity:.3f})")
                return best_match
            else:
                logging.warning(f"分类 '{field}' 与预定义分类相似度较低 (最高相似度: {max_similarity:.3f})，保留原分类")
                return field
                
        except Exception as e:
            logging.error(f"计算embedding相似度时出错: {e}，使用编辑距离作为备选")
            # 备选方案：使用编辑距离
            return self._fallback_normalize(field)
    
    def _fallback_normalize(self, field):
        """备选方案：使用编辑距离"""
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        min_distance = float('inf')
        best_match = field
        
        for category in PREDEFINED_CATEGORIES:
            distance = levenshtein_distance(field, category)
            if distance < min_distance:
                min_distance = distance
                best_match = category
        
        if min_distance <= 5:  # 编辑距离阈值
            logging.info(f"分类 '{field}' 被规范化为 '{best_match}' (编辑距离: {min_distance})")
            return best_match
        else:
            logging.warning(f"分类 '{field}' 与预定义分类距离较远 (最小距离: {min_distance})，保留原分类")
            return field

    def parse_llm_output(self, output_text):
        """使用正则表达式解析 LLM 的结构化输出"""
        papers = []
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
                    
                    # 使用embedding距离规范化研究方向分类
                    raw_field = field.group(1).strip()
                    paper_data['field'] = self.normalize_field(raw_field)
                    
                    papers.append(paper_data)
                else:
                    logging.warning(f"解析论文块失败，缺少字段: {block}")
            except Exception as e:
                logging.warning(f"解析论文块时出现异常: {e}\n块内容: {block}")
                
        return papers

    def group_and_sort_papers(self, papers):
        """对解析后的论文列表进行排序和分组"""
        papers.sort(key=lambda x: x['score'], reverse=True)
        
        grouped_papers = {}
        for paper in papers:
            field = paper.get('field', '其他')
            if field not in grouped_papers:
                grouped_papers[field] = []
            grouped_papers[field].append(paper)
            
        return grouped_papers

    def process_papers(self):
        """处理论文的主要逻辑"""
        logging.info("--- 开始执行每日 ArXiv 推荐任务 ---")
        
        # 1. 获取所有分类的论文
        all_papers = []
        for category in CATEGORIES_TO_FETCH:
            all_papers.extend(self.get_recent_papers(category, RSS_FILTER_KEY))
        
        # 去重
        seen_links = set()
        unique_papers = []
        for paper in all_papers:
            if paper['link'] not in seen_links:
                unique_papers.append(paper)
                seen_links.add(paper['link'])
        
        total_papers_read = len(unique_papers)
        logging.info(f"共获取到 {total_papers_read} 篇不重复的新论文。")
        
        # 保存原始论文列表到 RAW.md
        raw_md_content = self.list_to_raw_markdown(unique_papers)
        self.write_file("RAW.md", raw_md_content)
        
        if total_papers_read == 0:
            logging.info("未获取到新论文，任务提前结束。")
            return {
                'grouped_papers': {},
                'total_papers_read': 0,
                'total_tokens_used': 0,
                'update_time': datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
            }

        # 2. 分批处理
        all_llm_outputs = []
        total_tokens_used = 0
        batch_input_md_content = "" # 用于保存发给 LLM 的原始输入
        
        num_batches = math.ceil(total_papers_read / BATCH_SIZE)
        logging.info(f"开始分批处理，共 {num_batches} 批，每批 {BATCH_SIZE} 篇。")
        
        for i in range(num_batches):
            batch_start = i * BATCH_SIZE
            batch_end = (i + 1) * BATCH_SIZE
            batch_papers = unique_papers[batch_start:batch_end]
            
            logging.info(f"正在处理第 {i+1}/{num_batches} 批...")
            
            # 准备输入
            batch_llm_input_md = self.list_to_markdown_for_llm(batch_papers)
            batch_input_md_content += f"# --- 批次 {i+1}/{num_batches} ---\n\n{batch_llm_input_md}\n\n"
            
            query = self.prompt_template.format(
                batch_num=i + 1,
                total_batches=num_batches,
                paper_list=batch_llm_input_md
            )
            
            # 调用 LLM
            response_content, tokens = self.get_llm_response(query)
            total_tokens_used += tokens
            
            if "[ERROR:" not in response_content and "[NO_RELEVANT_PAPERS]" not in response_content:
                all_llm_outputs.append(response_content)
            elif "[NO_RELEVANT_PAPERS]" in response_content:
                logging.info(f"第 {i+1} 批没有相关论文")
                
        logging.info(f"所有批次处理完毕。总消耗 Tokens: {total_tokens_used}")
        
        # 写入批次输入内容（用于调试）
        self.write_file("BATCH_INPUT.md", batch_input_md_content)

        # 3. 解析和后处理
        full_llm_output = "\n".join(all_llm_outputs)
        parsed_papers = self.parse_llm_output(full_llm_output)
        logging.info(f"从 LLM 输出中成功解析出 {len(parsed_papers)} 篇相关论文。")
        
        grouped_and_sorted_papers = self.group_and_sort_papers(parsed_papers)
        
        return {
            'grouped_papers': grouped_and_sorted_papers,
            'total_papers_read': total_papers_read,
            'total_tokens_used': total_tokens_used,
            'update_time': datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
            'parsed_papers': parsed_papers
        }

    def write_file(self, filename, content):
        """安全的写入文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"成功写入文件: {filename}")
        except IOError as e:
            logging.error(f"写入文件 {filename} 时出错: {e}")

    def cleanup_old_history(self):
        """清理超过 MAX_HISTORY_DAYS 天的历史文件"""
        try:
            if not os.path.exists(HISTORY_DIR):
                return
                
            current_date = datetime.now()
            for filename in os.listdir(HISTORY_DIR):
                if filename.endswith('.html') or filename.endswith('.md'):
                    # 从文件名提取日期
                    date_str = filename.replace('.html', '').replace('.md', '')
                    try:
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        # 计算天数差
                        days_diff = (current_date - file_date).days
                        if days_diff > MAX_HISTORY_DAYS:
                            file_path = os.path.join(HISTORY_DIR, filename)
                            os.remove(file_path)
                            logging.info(f"删除过期历史文件: {filename}")
                    except ValueError:
                        logging.warning(f"跳过无法解析日期的文件: {filename}")
        except Exception as e:
            logging.error(f"清理历史文件时出错: {e}")

    def get_history_dates(self):
        """获取所有历史文件的日期列表"""
        history_dates = []
        try:
            if os.path.exists(HISTORY_DIR):
                for filename in os.listdir(HISTORY_DIR):
                    if filename.endswith('.html'):
                        date_str = filename.replace('.html', '')
                        try:
                            # 验证日期格式
                            datetime.strptime(date_str, '%Y-%m-%d')
                            history_dates.append(date_str)
                        except ValueError:
                            continue
                history_dates.sort(reverse=True)  # 按日期降序排列
        except Exception as e:
            logging.error(f"获取历史日期列表时出错: {e}")
        
        return history_dates