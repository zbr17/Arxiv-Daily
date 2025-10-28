import os
import logging
from arxiv_daily import ArXivProcessor
from web_generator import WebGenerator

# 配置项
HISTORY_DIR = "./history"
MODEL_ID = "doubao-seed-1-6-thinking-250715"

def main():
    logging.info("=== ArXiv 每日推荐系统启动 ===")
    
    try:
        # 初始化处理器和生成器
        processor = ArXivProcessor()
        web_generator = WebGenerator(MODEL_ID)
        
        # 处理论文
        result = processor.process_papers()
        
        # 获取历史日期列表（在处理前获取，避免包含今天的新文件）
        history_dates = processor.get_history_dates()
        
        # 生成 README.md
        readme_content = web_generator.generate_readme(
            result['grouped_papers'], 
            result['total_papers_read'], 
            result['total_tokens_used'], 
            result['update_time']
        )
        processor.write_file("README.md", readme_content)
        
        # 生成历史 HTML 文件
        date_str = result['update_time'][:10]
        history_filename = f"{HISTORY_DIR}/{date_str}.html"
        history_html_content = web_generator.generate_html(
            result['grouped_papers'],
            result['total_papers_read'],
            result['total_tokens_used'],
            result['update_time'],
            history_dates,
            is_index=False  # 历史文件不需要历史选择器
        )
        processor.write_file(history_filename, history_html_content)
        
        # 生成历史 Markdown 文件
        history_md_filename = f"{HISTORY_DIR}/{date_str}.md"
        history_md_content = web_generator.generate_history_markdown(
            result['grouped_papers'],
            result['total_papers_read'],
            result['total_tokens_used'],
            result['update_time']
        )
        processor.write_file(history_md_filename, history_md_content)
        
        # 生成主 index.html（包含历史选择器）
        index_html_content = web_generator.generate_html(
            result['grouped_papers'],
            result['total_papers_read'],
            result['total_tokens_used'],
            result['update_time'],
            history_dates,
            is_index=True  # 主页需要历史选择器
        )
        processor.write_file("index.html", index_html_content)
        
        # 清理过期历史文件
        processor.cleanup_old_history()
        
        logging.info("=== 任务执行完毕 ===")
        
    except Exception as e:
        logging.error(f"执行过程中发生错误: {e}")
        # 生成错误页面
        web_generator = WebGenerator(MODEL_ID)
        error_content = web_generator.generate_html(
            {}, 0, 0, 
            f"系统错误: {str(e)}", 
            []
        )
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(error_content)
        raise

if __name__ == "__main__":
    main()