import os
import requests
import pandas as pd
import time
import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict

# ====== APIキー設定（環境変数から取得） ======
API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemini-2.0-flash"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY
)

# ====== ニュースソース ======
NEWS_SOURCES = {
    "ITmedia_AI": "https://rss.itmedia.co.jp/rss/2.0/aiplus.xml",
}

# ====== 広告除外キーワード ======
AD_KEYWORDS = ["PR", "広告", "sponsored", "スポンサー", "タイアップ"]
AD_URL_PATTERNS = ["/ad/", "/special/", "/sponsor/", "/pr/"]

# ====== 状態定義 ======
class NewsState(TypedDict):
    df_news: object
    df_summarized: object
    report: str

# ====== ノード① RSS収集 ======
def get_news_articles():
    news_data = []
    for category, url in NEWS_SOURCES.items():
        print(f"🔍 {category} のニュースを取得中...")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "xml")
        articles = soup.find_all("item")[:3]
        if not articles:
            print(f"⚠️ {category} カテゴリでニュースが見つかりませんでした。")
        for article in articles:
            title_element = article.find('title')
            title = title_element.text if title_element else 'No Title'

            link_element = article.find('link')
            link = link_element.text if link_element else 'No Link'

            pub_date_element = article.find('pubDate')
            pub_date = pub_date_element.text if pub_date_element else 'No Date'

            description_element = article.find('description')
            description = description_element.text if description_element else ''

            is_ad = any(kw.lower() in title.lower() for kw in AD_KEYWORDS) or \
                    any(pt in link for pt in AD_URL_PATTERNS)
            if is_ad:
                print(f"⏭️ 広告スキップ: {title}")
                continue

            news_data.append([category, title, link, pub_date, description])
            time.sleep(1)

    df = pd.DataFrame(news_data, columns=["カテゴリ", "タイトル", "URL", "公開日", "概要"])
    print("✅ ニュース取得処理完了！")
    return df

def collect_node(state: NewsState) -> dict:
    print("📡 ニュース収集中...")
    df = get_news_articles()
    return {"df_news": df}

# ====== ノード② 要約 ======
def summarize_news(df):
    summaries = []
    print("要約を開始します...")
    for index, row in df.iterrows():
        title = row["タイトル"]
        description = row["概要"]

        summary_prompt = f"""以下の記事を200文字以内で日本語で要約してください。

タイトル：{title}
概要：{description}"""

        summary_response = genai.GenerativeModel(MODEL_NAME).generate_content(summary_prompt)
        summary = summary_response.text.strip()
        summaries.append(summary)
        print(f"  ✅ 要約完了：{title[:20]}...")
        time.sleep(10)

    print("✅ ニュース要約完了！")
    df["要約"] = summaries
    return df

def summarize_node(state: NewsState) -> dict:
    print("📝 要約中...")
    df = summarize_news(state["df_news"].copy())
    return {"df_summarized": df}

# ====== ノード③ レポート生成 ======
def report_node(state: NewsState) -> dict:
    print("📊 レポート生成中...")
    df = state["df_summarized"]
    news_text = "\n".join([
        f"・{row['タイトル']}（{row['カテゴリ']}）\n  {row['要約']}"
        for _, row in df.iterrows()
    ])
    prompt = f"""以下のニュース一覧から、SIer新入社員向けの日報を作成してください。

{news_text}

形式：
# IT日報
## 今日の重要ニュースTOP3
（タイトル・なぜ重要か・一言コメント）
## 今日のキーワード
（3つ）
## まとめ
（2文）"""
    report = llm.invoke(prompt).content
    return {"report": report}

# ====== グラフ組み立て ======
graph = StateGraph(NewsState)
graph.add_node("collect",   collect_node)
graph.add_node("summarize", summarize_node)
graph.add_node("report",    report_node)

graph.set_entry_point("collect")
graph.add_edge("collect",   "summarize")
graph.add_edge("summarize", "report")
graph.add_edge("report",    END)

app = graph.compile()

# ====== 実行 ======
print("🚀 ITニュースエージェント起動...\n")
result = app.invoke({
    "df_news": None,
    "df_summarized": None,
    "report": ""
})

# レポート表示
print("\n" + "="*50)
print(result["report"])
print("="*50)

# CSV保存（GitHub ActionsではExcelより軽いCSVがおすすめ）
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result["df_summarized"].to_csv(
    f"news_{current_time}.csv", index=False, encoding="utf-8-sig"
)
print(f"📁 CSV保存しました: news_{current_time}.csv")
