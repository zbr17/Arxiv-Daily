import os
import re
from datetime import datetime

class WebGenerator:
    def __init__(self, model_id):
        self.model_id = model_id
        
    def generate_readme(self, grouped_papers, total_papers_read, total_tokens, update_time):
        """生成 README.md 文件的内容"""
        
        md_content = f"# ArXiv 每日推荐\n\n"
        md_content += f"> 更新于北京时间：{update_time}\n"
        md_content += f"> 已自动阅读了 {total_papers_read} 篇最新的论文。\n"
        md_content += f"> 使用模型：{self.model_id} | 消耗 Tokens：{total_tokens}\n\n"
        
        if not grouped_papers:
            md_content += "今日未发现相关论文。\n"
            return md_content

        # 添加导航
        md_content += "## 快速导航\n\n"
        for field in grouped_papers:
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

    def generate_history_markdown(self, grouped_papers, total_papers_read, total_tokens, update_time):
        """生成历史 Markdown 文件的内容"""
        
        date_str = update_time[:10]  # 提取日期部分
        md_content = f"# ArXiv 每日推荐 - {date_str}\n\n"
        md_content += f"> 更新于北京时间：{update_time}\n"
        md_content += f"> 已自动阅读了 {total_papers_read} 篇最新的论文。\n"
        md_content += f"> 使用模型：{self.model_id} | 消耗 Tokens：{total_tokens}\n\n"
        
        if not grouped_papers:
            md_content += "今日未发现相关论文。\n"
            return md_content

        # 生成论文列表
        for field, papers in grouped_papers.items():
            md_content += f"## {field}\n\n"
            for paper in papers:
                md_content += f"### [Score: {paper['score']}/10] {paper['title']}\n"
                md_content += f"- **Authors:** {paper['authors']}\n"
                md_content += f"- **Published:** {paper['published']}\n"
                md_content += f"- **Link:** [{paper['link']}]({paper['link']})\n"
                md_content += f"- **Reason:** {paper['reason']}\n\n"
                
        return md_content

    def generate_html(self, grouped_papers, total_papers_read, total_tokens, update_time, history_dates=None, is_index=False):
        """
        生成 HTML 文件的内容
        history_dates: 历史日期列表，用于生成下拉菜单
        is_index: 是否是主页，主页需要显示历史选择器
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
            .field-section { padding-top: 60px; margin-top: -60px; }
            .history-selector { margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .history-selector label { font-weight: bold; margin-right: 10px; }
            .history-selector select { padding: 5px 10px; border: 1px solid #ddd; border-radius: 3px; }
        </style>
        """

        # HTML 头部
        html_content = f"<!DOCTYPE html>\n<html lang='zh-CN'>\n<head>\n"
        html_content += f"<meta charset='UTF-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
        html_content += f"<title>ArXiv 每日推荐 - {update_time[:10]}</title>\n{css_style}\n</head>\n<body>\n"

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
        html_content += f"<h1>ArXiv 每日推荐 - {update_time[:10]}</h1>\n"
        
        # 历史记录选择器 - 只在主页显示
        if is_index and history_dates:
            html_content += "<div class='history-selector'>\n"
            html_content += "<label for='history-date'>选择历史日期:</label>\n"
            html_content += "<select id='history-date' onchange='onHistoryDateChange(this)'>\n"
            html_content += f"<option value='index.html'>最新 ({update_time[:10]})</option>\n"
            for date in history_dates:
                # 确保日期不是今天
                if date != update_time[:10]:
                    html_content += f"<option value='history/{date}.html'>{date}</option>\n"
            html_content += "</select>\n"
            html_content += "</div>\n"

        html_content += f"<div class='meta-info'>"
        html_content += f"<p>更新于北京时间：{update_time}</p>\n"
        html_content += f"<p>已自动阅读了 {total_papers_read} 篇最新的论文。</p>\n"
        html_content += f"<p>使用模型：{self.model_id} | 消耗 Tokens：{total_tokens}</p>\n"
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
        
        # JavaScript
        html_content += """
        <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            const sections = document.querySelectorAll('.field-section');
            const navLinks = document.querySelectorAll('.navbar a');

            function changeLinkState() {
                let index = sections.length;

                while(--index && window.scrollY + 100 < sections[index].offsetTop) {}

                navLinks.forEach((link) => link.classList.remove('active'));
                if (navLinks[index]) {
                    navLinks[index].classList.add('active');
                }
            }

            changeLinkState();
            window.addEventListener('scroll', changeLinkState);
        });

        function onHistoryDateChange(select) {
            if (select.value) {
                window.location.href = select.value;
            }
        }
        </script>
        """
        html_content += "</body>\n</html>"
        return html_content