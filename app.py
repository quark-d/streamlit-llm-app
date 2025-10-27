# app.py
# Lesson21 Chapter6 提出課題：StreamlitでLLMアプリ
# 条件：
# - 入力フォーム1つ + ラジオボタンで専門家ロールを切替
# - LangChainでLLMを呼び出し
# - （入力テキスト, ラジオ選択値）を引数に取り、LLM回答を返す関数を定義
# - 使い方テキストを画面に表示
# - Python 3.11 を想定

import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# =====================
# 1) APIキーの読み込み
# =====================
# ローカル（.env）
load_dotenv()

# =====================
# 2) 専門家ロール定義
# =====================
ROLES = {
    "採用アドバイザー": {
        "system": (
            "あなたは現場に強い採用アドバイザーです。 "
            "求人票改善、選考プロセス、候補者体験の観点から、実務で今すぐ試せる施策を優先度付きで提示してください。"
        )
    },
    "Pythonメンター": {
        "system": (
            "あなたは優しく厳密なPythonメンターです。 "
            "コード例・注意点・ベストプラクティスを示し、短いサンプルとチェックリストも付けて説明してください。"
        )
    },
}

# =====================
# 3) LLM呼び出し関数
# =====================
#  必須条件： (入力テキスト, ラジオ選択値) → 回答文字列

def ask_llm(user_text: str, role_key: str) -> str:
    """選択した専門家ロールのシステムメッセージとユーザー入力をもとに、LLMの回答を返す。"""
    if not user_text.strip():
        return "（入力が空です）"

    system_msg = ROLES[role_key]["system"]

    # LangChainのプロンプトテンプレート（Lesson8)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = prompt | llm  # Runnableチェーン
    result = chain.invoke({"input": user_text})
    return result.content if hasattr(result, "content") else str(result)

# =====================
# 4) Streamlit UI
# =====================
st.set_page_config(page_title="LLM Webアプリ（Lesson21 提出課題）", page_icon="🤖")

st.title("LLM Web アプリ 🧪")
st.caption(
    "Lesson21 Chapter6 提出課題サンプル：\n入力とロール選択からLLMに質問し、結果を表示します。"
)

with st.expander("このアプリの使い方 / 注意事項", expanded=True):
    st.markdown(
        """
        **使い方**
        1. 右（または下）の *専門家ロール* から目的に合うロールを選びます。
        2. 入力欄に質問や相談内容を記入し、**送信**を押します。
        3. 画面下部にLLMの回答が表示されます。

        **注意**
        - 本サンプルの回答は参考情報です。重要な意思決定は一次情報で必ず検証してください。
        """
    )

# サイドバー：ロール選択
st.sidebar.header("設定")
role_choice = st.sidebar.radio("専門家ロール", list(ROLES.keys()), index=0)

# 入力フォーム（1つ）
with st.form("ask_form", clear_on_submit=False):
    user_text = st.text_area("質問 / 相談内容を入力", height=160, placeholder="質問内容を入力します")
    submitted = st.form_submit_button("送信")

# 送信処理
if submitted:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI APIキーが設定されていません。.env または Secrets を確認してください。")
    else:
        with st.spinner("LLMに問い合わせ中…"):
            answer = ask_llm(user_text, role_choice)
        st.markdown("---")
        st.subheader("回答")
        st.write(answer)

