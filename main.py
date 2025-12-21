from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 部署专用
from fastapi.responses import FileResponse   # 部署专用
from pydantic import BaseModel
import requests
import uvicorn
import re
import os
from collections import Counter, defaultdict
from snownlp import SnowNLP
from textblob import TextBlob
from langdetect import detect, LangDetectException
import jieba
import jieba.posseg as pseg
import nltk
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- 0. 环境初始化 ---
def download_nltk_data():
    required = [
        ('tokenizers/punkt', 'punkt'), 
        ('tokenizers/punkt_tab', 'punkt_tab'), 
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'), 
        ('corpora/stopwords', 'stopwords')
    ]
    print("⏳ 初始化 NLP 学术组件...")
    for path, name in required:
        try: nltk.data.find(path)
        except LookupError:
            try: nltk.download(name, quiet=True)
            except: pass
download_nltk_data()

app = FastAPI()

# 允许跨域（本地开发方便，部署后也无妨）
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- 1. 停用词库 (保持 V13/V14 的高标准) ---
BASIC_BLOCKLIST = {'could', 'would', 'should', 'can', 'will', 'must', 'might', 'want', 'wanted', 'wants', 'need', 'needs', 'also', 'still', 'even', 'just', 'only', 'really', 'actually', 'already', 'every', 'each', 'another', 'other', 'any', 'some', 'many', 'much', 'well', 'very', 'pretty', 'quite', 'too', 'so', 'first', 'second', 'next', 'last', 'finally', 'gameplay', 'game', 'games', 'gaming', 'player', 'players', 'steam', 'review', 'recommend', 'pc', 'version', 'good', 'bad', 'great', 'nice', 'fun', 'love', 'like', 'best', 'better', 'people', 'person', 'guy', 'guys', 'man', 'men', 'woman', '1010', 'rate', 'rating', 'story', 'map', 'action'}
ABSTRACT_NOUNS = {'thing', 'things', 'stuff', 'bit', 'lot', 'way', 'kind', 'sort', 'type', 'part', 'side', 'point', 'reason', 'example', 'idea', 'opinion', 'thought', 'issue', 'problem', 'matter', 'case', 'fact', 'moment', 'time', 'hour', 'hours', 'minute', 'minutes', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years', 'look', 'feel', 'use', 'try', 'start', 'end', 'run', 'work', 'job', 'attempt', 'change', 'chance', 'nothing', 'anything', 'something', 'everything', 'someone', 'anyone', 'everyone'}
FINAL_ENGLISH_BLOCKLIST = BASIC_BLOCKLIST | ABSTRACT_NOUNS
CHINESE_BLACKLIST = {'只能', '还有', '没有', '无需', '需要', '可能', '确实', '其实', '甚至', '第一', '第二', '最后', '首先', '其次', '之前', '之后', '现在', '目前', '让玩家', '玩家', '游戏', '游玩', '内容', '体验', '感觉', '觉得', '一点', '有点', '一些', '一样', '一下', '一次', '个', '次', '种', '什么', '怎么', '为什么', '因为', '所以', '但是', '虽然', '不过', '以及', '并且', '而且', '真的', '非常', '特别', '相当', '比较', '太', '挺', '蛮', '好玩', '推荐', '差评', '神作', '时候', '时间', '小时', '分钟', '时长', '东西', '地方', '部分', '方面', '程度', '情况', '问题', '原因', '例子'}
JAPANESE_STOPWORDS = {'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう', 'また', 'もの', 'という', 'あり', 'まで', 'られ', 'それ', 'みる', 'だけ', 'これ', 'でき', 'ます', 'ので', 'ゲーム', 'プレイ', 'おすすめ', 'レビュー', '面白い', '楽しい', 'ストーリー', 'マップ', 'アクション'}
try:
    from nltk.corpus import stopwords
    FINAL_ENGLISH_BLOCKLIST |= set(stopwords.words('english'))
except: pass

# --- 2. 核心处理逻辑 ---
def check_language(text, target_lang):
    if not text or len(text) < 5: return False
    try:
        detected = detect(text)
        if target_lang == 'english': return detected == 'en'
        elif target_lang == 'schinese': return detected.startswith('zh') or len(re.findall(r'[\u4e00-\u9fa5]', text)) / len(text) > 0.3
        elif target_lang == 'japanese': return detected == 'ja' or len(re.findall(r'[\u3040-\u30ff]', text)) > 0 
        return True 
    except: return False

def basic_clean(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text, lang):
    try:
        if not text: return 0.5
        if lang in ['schinese', 'tchinese']: return SnowNLP(text).sentiments
        else: return (TextBlob(text).sentiment.polarity + 1) / 2
    except: return 0.5

def get_tokens(text, lang):
    tokens = []
    if lang in ['schinese', 'tchinese']:
        clean_text = re.sub(r'[a-zA-Z]+', '', text) 
        words = pseg.cut(clean_text)
        for w, flag in words:
            if len(w) < 2 or w in CHINESE_BLACKLIST: continue
            if flag in ['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'a', 'ag']:
                tokens.append(w)
    elif lang == 'japanese':
        clean_text = re.sub(r'[a-zA-Z]+', '', text)
        raw_tokens = re.findall(r'[\u4e00-\u9faf\u30a0-\u30ff]{2,}', clean_text)
        tokens = [w for w in raw_tokens if w not in JAPANESE_STOPWORDS]
    else:
        try:
            raw_words = nltk.word_tokenize(text.lower())
            tagged_words = nltk.pos_tag(raw_words)
            for w, tag in tagged_words:
                if len(w) <= 2 or w.isdigit() or w in FINAL_ENGLISH_BLOCKLIST: continue
                if tag.startswith('NN') or tag.startswith('JJ'):
                    tokens.append(w)
        except:
            raw_words = text.lower().split()
            tokens = [w for w in raw_words if len(w)>3 and not w.isdigit()]
    return tokens

# --- 3. LDA 智能分析 (含去重) ---
def perform_lda(reviews, lang, auto_topic=False, manual_k=5):
    corpus = []
    for r in reviews:
        tokens = get_tokens(r['content'], lang)
        if tokens:
            corpus.append(" ".join(tokens))
    
    if len(corpus) < 10: return []

    try:
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
        tf = tf_vectorizer.fit_transform(corpus)
        feature_names = tf_vectorizer.get_feature_names_out()

        best_lda = None
        best_k = manual_k

        if auto_topic:
            best_score = float('inf')
            for k in range(3, 8):
                lda = LatentDirichletAllocation(n_components=k, max_iter=5, learning_method='online', random_state=42)
                lda.fit(tf)
                perplexity = lda.perplexity(tf)
                if perplexity < best_score:
                    best_score = perplexity
                    best_lda = lda
                    best_k = k
        else:
            best_lda = LatentDirichletAllocation(n_components=manual_k, max_iter=10, learning_method='online', random_state=42)
            best_lda.fit(tf)

        topics = []
        used_words = set()

        for topic_idx, topic in enumerate(best_lda.components_):
            top_features_ind = topic.argsort()[:-15:-1] 
            candidate_words = [feature_names[i] for i in top_features_ind]
            
            clean_top_words = []
            for word in candidate_words:
                if word not in used_words:
                    clean_top_words.append(word)
                    used_words.add(word)
                if len(clean_top_words) >= 6:
                    break
            
            topics.append({
                "id": topic_idx + 1,
                "keywords": clean_top_words,
                "perplexity": round(best_lda.perplexity(tf), 2) if auto_topic else None
            })
        
        if auto_topic and topics:
            topics[0]['auto_k_result'] = best_k

        return topics
    except Exception as e:
        print(f"LDA Error: {e}")
        return []

def extract_keywords_counter(reviews, lang):
    all_tokens = []
    for r in reviews:
        all_tokens.extend(get_tokens(r['content'], lang))
    return [{"name": k, "value": v} for k, v in Counter(all_tokens).most_common(60)]

def calculate_trends(reviews):
    date_map = defaultdict(lambda: {'count': 0, 'total_sentiment': 0})
    for r in reviews:
        date_str = datetime.fromtimestamp(r['date']).strftime('%Y-%m-%d')
        date_map[date_str]['count'] += 1
        date_map[date_str]['total_sentiment'] += r['sentiment']
    trend_list = []
    for date, stats in date_map.items():
        trend_list.append({'date': date, 'count': stats['count'], 'avg_sentiment': round(stats['total_sentiment'] / stats['count'], 2)})
    trend_list.sort(key=lambda x: x['date'])
    return trend_list

# --- 4. API 路由定义 ---
class SearchRequest(BaseModel): keyword: str
class CountRequest(BaseModel): app_id: str; language: str
class ScrapeRequest(BaseModel): app_id: str; language: str; count: int; clean_mode: bool = False; lda_auto: bool = False; lda_k: int = 5

@app.post("/api/search")
def search_api(req: SearchRequest):
    try:
        r = requests.get("https://store.steampowered.com/api/storesearch/", params={'term': req.keyword, 'l': 'english', 'cc': 'US'}).json()
        if r['total'] > 0: return {"name": r['items'][0]['name'], "id": r['items'][0]['id'], "img": r['items'][0]['tiny_image']}
        raise HTTPException(404, "Not found")
    except: raise HTTPException(404, "Error")

@app.post("/api/check_count")
def check_count(req: CountRequest):
    try:
        r = requests.get(f"https://store.steampowered.com/appreviews/{req.app_id}", params={'json': 1, 'language': req.language, 'num_per_page': 0}).json()
        return {"total": r['query_summary'].get('total_reviews', 0) if 'query_summary' in r else 0}
    except: return {"total": 0}

@app.post("/api/scrape")
def scrape_api(req: ScrapeRequest):
    reviews = []
    cursor = '*'
    url = f"https://store.steampowered.com/appreviews/{req.app_id}"
    request_limit = req.count * 3 if req.count > 0 else 10000
    
    while len(reviews) < req.count:
        try:
            resp = requests.get(url, params={'json': 1, 'filter': 'recent', 'language': req.language, 'review_type': 'all', 'purchase_type': 'all', 'num_per_page': 100, 'cursor': cursor}, timeout=10)
            data = resp.json()
            if 'reviews' in data and len(data['reviews']) > 0:
                for r in data['reviews']:
                    if len(reviews) >= req.count: break
                    cleaned = basic_clean(r['review'].replace('\n', ' '))
                    if req.clean_mode and not check_language(cleaned, req.language): continue
                    if not cleaned: continue
                    reviews.append({
                        'author_id': r['author']['steamid'],
                        'playtime': round(r['author']['playtime_forever']/60, 1),
                        'content': cleaned,
                        'votes_up': r['votes_up'],
                        'sentiment': round(analyze_sentiment(cleaned, req.language), 2),
                        'date': r['timestamp_created']
                    })
                cursor = data['cursor']
                if len(reviews) + len(data['reviews']) > request_limit: break
            else: break
        except: break

    s_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in reviews:
        if r['sentiment'] > 0.6: s_counts['positive'] += 1
        elif r['sentiment'] < 0.4: s_counts['negative'] += 1
        else: s_counts['neutral'] += 1
        
    return {
        "count": len(reviews), 
        "data": reviews, 
        "analysis": {
            "sentiment": s_counts, 
            "keywords": extract_keywords_counter(reviews, req.language),
            "trends": calculate_trends(reviews),
            "lda_topics": perform_lda(reviews, req.language, req.lda_auto, req.lda_k) 
        }
    }

# --- 5. 部署关键代码 ---
# 挂载静态资源
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路由返回 index.html
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    # 监听 0.0.0.0 以允许外部访问
    uvicorn.run(app, host="0.0.0.0", port=8000)