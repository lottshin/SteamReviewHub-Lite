from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import uvicorn
import re
from collections import Counter, defaultdict
from datetime import datetime
import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- NEXUS 全功能版 (针对 Zeabur 优化内存) ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- 1. 停用词库 (精简版) ---
# 这里的词在做 LDA 时会被过滤掉，保证主题质量
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'it', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'with', 'as', 'for', 'game', 'play', 'player', 'games', 'review', 'steam', 'recommend', 'fun', 'good', 'bad', 'great', 'very', 'just', 'can', 'so', 'really', 'get', 'like', 'time', 'play', 'would', 'make', 'story', 'one', 'much', 'even', 'best', 'better', 'played', 'playing', 'hours', '10', '100', 'feels', 'feeling', 'lot', 'bit', 'way', 'thing', 'things', 'stuff', 'people', 'person', 'man', 'guy', 'guys', 'recommendation', 'simulator', 'access', 'early', 'version', 'now', 'still', 'know', 'think', 'see', 'say', 'go', 'come', 'take', 'look', 'want', 'need', 'use', 'find', 'give', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'chinese', 'english', 'japanese', 'korean', 'russian', 'spanish', 'french', 'german', 'italian', 'portuguese', 'brazilian', 'polish', 'turkish', 'thai', 'vietnamese', 'dutch', 'swedish', 'norwegian', 'danish', 'finnish', 'hungarian', 'czech', 'romanian', 'ukrainian', 'greek', 'bulgarian', 'arabic', 'hebrew', 'hindi', 'indonesian', 'malay', 'filipino'}
CHINESE_STOP = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '游戏', '玩家', '玩', '推荐', '体验', '感觉', '觉得', '不错', '非常', '真的', '还是', '就是', '这个', '那个', '虽然', '但是', '因为', '所以', '如果', '比如', '不仅', '而且', '或者', '不过', '甚至', '其实', '可能', '不仅', '还有', '并且', '以及', '然后', '于是', '不仅', '哪怕', '或是', '要么', '与其', '宁可', '不仅', '不光', '不单', '单单', '只', '光', '仅', '仅仅', '就', '只', '仅', '光', '才', '再', '又', '也', '还', '犹', '尚', '仍', '更', '越', '越发', '更加', '越加', '愈', '愈发', '愈加', '特', '特别', '尤其', '尤其', '分外', '格外', '相当', '很', '太', '极', '最', '十分', '非常', '异常', '顶', '极度', '极端', '至', '至多', '顶多', '最多', '至少', '最少', '起码', '稍微', '稍稍', '多少', '多', '少', '略', '略微', '有点', '有点儿', '有些', '一些'}

def clean_text(text, lang):
    text = re.sub(r'<[^>]+>', '', text) # 去HTML标签
    text = re.sub(r'http\S+', '', text) # 去链接
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = []
    if lang in ['schinese', 'tchinese']:
        # 中文使用 Jieba
        words = jieba.cut(text)
        tokens = [w for w in words if len(w) > 1 and w not in CHINESE_STOP and not re.match(r'^[a-zA-Z0-9]+$', w)]
    else:
        # 英文简单分词
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        tokens = [w for w in words if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]
    
    return tokens

# --- 2. 核心功能：情感分析 & LDA & 趋势 ---

def get_sentiment(text, lang):
    try:
        if lang in ['schinese', 'tchinese']:
            return SnowNLP(text).sentiments # 0~1 之间
        else:
            # 简单的英文词典匹配 fallback
            # (SnowNLP 其实也可以处理英文，但效果一般，这里为了省内存不引 TextBlob)
            return 0.5 
    except:
        return 0.5

def perform_lda(reviews_tokens, manual_k=5):
    if len(reviews_tokens) < 10: return []
    
    # 重新组合成句子给 sklearn 用
    corpus = [" ".join(tokens) for tokens in reviews_tokens if tokens]
    
    if not corpus: return []

    try:
        # 关键优化：max_features=1000 限制内存占用
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
        tf = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # 关键优化：max_iter=5 减少计算时间，防止超时
        lda = LatentDirichletAllocation(n_components=manual_k, max_iter=5, learning_method='online', random_state=42)
        lda.fit(tf)

        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[:-7:-1] # 取前6个词
            keywords = [feature_names[i] for i in top_indices]
            topics.append({"id": topic_idx + 1, "keywords": keywords})
        
        return topics
    except Exception as e:
        print(f"LDA Error: {e}")
        return []

def calculate_trends(reviews):
    # 按日期聚合评论数和情感
    date_map = defaultdict(lambda: {'count': 0, 'total_sentiment': 0})
    
    for r in reviews:
        # 转换时间戳为日期字符串
        dt = datetime.fromtimestamp(r['date']).strftime('%Y-%m-%d')
        date_map[dt]['count'] += 1
        date_map[dt]['total_sentiment'] += r['sentiment']
    
    trend_list = []
    for date, stats in date_map.items():
        avg_sent = round(stats['total_sentiment'] / stats['count'], 2)
        trend_list.append({
            'date': date, 
            'count': stats['count'], 
            'avg_sentiment': avg_sent
        })
    
    # 按日期排序
    trend_list.sort(key=lambda x: x['date'])
    return trend_list

# --- API 路由 ---

class SearchRequest(BaseModel): keyword: str
class ScrapeRequest(BaseModel): app_id: str; language: str; count: int; clean_mode: bool = False; lda_auto: bool = False; lda_k: int = 5

@app.post("/api/search")
def search_api(req: SearchRequest):
    try:
        r = requests.get("https://store.steampowered.com/api/storesearch/", params={'term': req.keyword, 'l': 'english', 'cc': 'US'}, timeout=5).json()
        if r['total'] > 0: return {"name": r['items'][0]['name'], "id": r['items'][0]['id'], "img": r['items'][0]['tiny_image']}
        raise HTTPException(404, "Not found")
    except: raise HTTPException(404, "Error")

@app.post("/api/scrape")
def scrape_api(req: ScrapeRequest):
    reviews = []
    cursor = '*'
    url = f"https://store.steampowered.com/appreviews/{req.app_id}"
    
    # 在 Zeabur 上可以适当放宽，但建议不要超过 500 条，否则前端等太久会超时
    safe_count = min(req.count, 500) 
    
    processed_docs_tokens = []

    while len(reviews) < safe_count:
        try:
            resp = requests.get(url, params={'json': 1, 'filter': 'recent', 'language': req.language, 'review_type': 'all', 'purchase_type': 'all', 'num_per_page': 100, 'cursor': cursor}, timeout=10)
            data = resp.json()
            
            if 'reviews' in data and len(data['reviews']) > 0:
                for r in data['reviews']:
                    if len(reviews) >= safe_count: break
                    
                    raw_text = r['review'].replace('\n', ' ')
                    tokens = clean_text(raw_text, req.language)
                    
                    # 只有当评论有实质内容时才加入
                    if len(tokens) > 2:
                        sentiment_score = 0.5
                        # 仅对中文做 SnowNLP，英文没装 TextBlob 暂时给 0.5
                        if req.language in ['schinese', 'tchinese']:
                            sentiment_score = get_sentiment(raw_text, req.language)
                        
                        processed_docs_tokens.append(tokens)
                        
                        reviews.append({
                            'author_id': r['author']['steamid'],
                            'playtime': round(r['author']['playtime_forever']/60, 1),
                            'content': raw_text, # 保留原文给前端展示
                            'votes_up': r['votes_up'],
                            'sentiment': round(sentiment_score, 2),
                            'date': r['timestamp_created']
                        })
                
                cursor = data['cursor']
            else: break
        except Exception as e: 
            print(f"Scrape loop error: {e}")
            break

    # 统计数据
    s_counts = {"positive": 0, "neutral": 0, "negative": 0}
    all_tokens = []
    
    for i, r in enumerate(reviews):
        # 重新统计情感
        if r['sentiment'] > 0.6: s_counts['positive'] += 1
        elif r['sentiment'] < 0.4: s_counts['negative'] += 1
        else: s_counts['neutral'] += 1
        
        # 收集词频
        all_tokens.extend(processed_docs_tokens[i])

    # 关键词 (词云数据)
    keywords = [{"name": k, "value": v} for k, v in Counter(all_tokens).most_common(60)]

    # 舆情演化趋势 (Trend)
    trends_data = calculate_trends(reviews)

    # LDA 主题模型
    lda_data = perform_lda(processed_docs_tokens, req.lda_k)

    return {
        "count": len(reviews), 
        "data": reviews, 
        "analysis": {
            "sentiment": s_counts, 
            "keywords": keywords,
            "trends": trends_data, # 趋势数据回归！
            "lda_topics": lda_data # LDA 数据回归！
        }
    }

# --- 部署配置 ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
