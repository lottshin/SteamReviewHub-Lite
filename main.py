from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import uvicorn
import re
from collections import Counter
from datetime import datetime

# --- 极简版 NEXUS (移除所有 AI 库以确保稳定运行) ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- 简单的分词函数 (不依赖 NLTK/Jieba) ---
def simple_tokenize(text, lang):
    # 移除标点和特殊字符
    text = re.sub(r'[^\w\s]', ' ', text)
    # 统一转小写
    words = text.lower().split()
    # 简单的停用词过滤 (硬编码一些最常见的)
    STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'it', 'this', 'that', 'for', 'with', 'as', 'by', 'game', 'play', 'player', 'review', 'games', 'really', 'very', 'just', 'can', 'so', 'much', 'not', 'have', 'has', 'had', 'be', 'do', 'does', 'did', 'but', 'from', 'all', 'we', 'my', 'your', 'me', 'up', 'out', 'if', 'about', 'get', 'like', 'good', 'bad', 'time', 'even', 'would', 'make', 'story', 'play', 'get', 'one', 'some', 'only', 'also', 'much', 'well', 'best', 'better', 'there', 'which', 'when', 'what', 'how', 'where', 'why', 'who', 'they', 'them', 'their', 'he', 'she', 'his', 'her', 'you', 'i', 'we', 'us', 'our', 'mine', 'yours', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'}
    # 中文简单按字切分 (凑合用，防止 Jieba 撑爆内存)
    if lang in ['schinese', 'tchinese']:
        return [char for char in text if '\u4e00' <= char <= '\u9fff']
    
    return [w for w in words if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]

# --- 简单的情感分析 (基于关键词，不依赖 SnowNLP/TextBlob) ---
# 这是一个非常简陋的替代方案，但能保证不崩溃
POSITIVE_WORDS = {'good', 'great', 'best', 'amazing', 'love', 'fun', 'nice', 'excellent', 'perfect', 'awesome', 'beautiful', 'fantastic', 'enjoy', 'recommend', '好', '棒', '强', '神作', '推荐', '喜欢', '不错', '优秀', '爽', '开心'}
NEGATIVE_WORDS = {'bad', 'worst', 'boring', 'hate', 'terrible', 'awful', 'trash', 'sucks', 'poor', 'bug', 'crash', 'fail', 'disappointed', '差', '烂', '垃圾', '不好', '无聊', '坑', 'bug', '卡顿', '失望'}

def simple_sentiment(text):
    text_lower = text.lower()
    score = 0.5
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    
    if pos_count > neg_count: score = 0.8
    elif neg_count > pos_count: score = 0.2
    return score

class SearchRequest(BaseModel): keyword: str
class CountRequest(BaseModel): app_id: str; language: str
class ScrapeRequest(BaseModel): app_id: str; language: str; count: int; clean_mode: bool = False; lda_auto: bool = False; lda_k: int = 5

@app.post("/api/search")
def search_api(req: SearchRequest):
    try:
        r = requests.get("https://store.steampowered.com/api/storesearch/", params={'term': req.keyword, 'l': 'english', 'cc': 'US'}, timeout=5).json()
        if r['total'] > 0: return {"name": r['items'][0]['name'], "id": r['items'][0]['id'], "img": r['items'][0]['tiny_image']}
        raise HTTPException(404, "Not found")
    except: raise HTTPException(404, "Error")

@app.post("/api/check_count")
def check_count(req: CountRequest):
    try:
        r = requests.get(f"https://store.steampowered.com/appreviews/{req.app_id}", params={'json': 1, 'language': req.language, 'num_per_page': 0}, timeout=5).json()
        return {"total": r['query_summary'].get('total_reviews', 0) if 'query_summary' in r else 0}
    except: return {"total": 0}

@app.post("/api/scrape")
def scrape_api(req: ScrapeRequest):
    reviews = []
    cursor = '*'
    url = f"https://store.steampowered.com/appreviews/{req.app_id}"
    safe_count = min(req.count, 300) # 强制限制最大 300 条
    
    while len(reviews) < safe_count:
        try:
            # 增加 timeout 到 15 秒
            resp = requests.get(url, params={'json': 1, 'filter': 'recent', 'language': req.language, 'review_type': 'all', 'purchase_type': 'all', 'num_per_page': 100, 'cursor': cursor}, timeout=15)
            data = resp.json()
            if 'reviews' in data and len(data['reviews']) > 0:
                for r in data['reviews']:
                    if len(reviews) >= safe_count: break
                    clean_text = re.sub(r'<[^>]+>', '', r['review']).replace('\n', ' ').strip()
                    if not clean_text: continue
                    
                    reviews.append({
                        'author_id': r['author']['steamid'],
                        'playtime': round(r['author']['playtime_forever']/60, 1),
                        'content': clean_text,
                        'votes_up': r['votes_up'],
                        'sentiment': simple_sentiment(clean_text), # 使用极简版情感分析
                        'date': r['timestamp_created']
                    })
                cursor = data['cursor']
                if len(reviews) + len(data['reviews']) > safe_count + 100: break
            else: break
        except Exception as e: 
            print(f"Scrape Error: {e}")
            break

    # 简单统计
    s_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in reviews:
        if r['sentiment'] > 0.6: s_counts['positive'] += 1
        elif r['sentiment'] < 0.4: s_counts['negative'] += 1
        else: s_counts['neutral'] += 1
    
    # 关键词统计
    all_words = []
    for r in reviews:
        all_words.extend(simple_tokenize(r['content'], req.language))
    keywords = [{"name": k, "value": v} for k, v in Counter(all_words).most_common(50)]

    return {
        "count": len(reviews), 
        "data": reviews, 
        "analysis": {
            "sentiment": s_counts, 
            "keywords": keywords,
            "trends": [], # 趋势图暂时留空以节省运算
            "lda_topics": [] # 彻底禁用 LDA
        }
    }

# --- 部署配置 ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
@app.head("/")
async def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
