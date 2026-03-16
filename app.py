import streamlit as st
import pandas as pd
import json
import re
import openai

# ==================== 工具函数 ====================

def read_csv_with_encoding(file):
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin1']
    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except:
            continue
    raise ValueError("无法读取 CSV 文件")


def detect_languages(text):
    """
    检测用户输入中的目标语言，支持上下文记忆
    """
    keywords = {
        "英语": ["英语", "英文", "english"],
        "日语": ["日语", "日文", "japanese"],
        "韩语": ["韩语", "韩文", "korean"],
        "法语": ["法语", "法文", "french"],
        "德语": ["德语", "德文", "german"],
        "西班牙语": ["西班牙语", "spanish"],
        "俄语": ["俄语", "俄文", "russian"],
    }
    
    # 上下文相关关键词（表示"使用之前的语言"）
    context_keywords = [
        "继续", "上面提到的", "之前提到的", "刚才", "同样的", "一样的", "刚才说的", "之前说",
        "上面一样的语言", "上面一样的", "翻译成上面一样的", "翻译成上面一样的语言",
        "和上面一样", "跟上面一样", "同上", "相同语言", "一样语言"
    ]
    
    text_lower = text.lower()
    
    # 1. 首先尝试检测具体语言
    detected = [lang for lang, keys in keywords.items() if any(k in text_lower for k in keys)]
    
    if detected:
        return detected, False  # False 表示不是上下文引用
    
    # 2. 如果没有检测到具体语言，检查是否是上下文引用
    is_context_reference = any(k in text for k in context_keywords)
    
    if is_context_reference:
        # 尝试从 session_state 获取之前的目标语言
        if "last_langs" in st.session_state and st.session_state.last_langs:
            return st.session_state.last_langs, True  # True 表示是上下文引用
        
        # 如果没有当前语言，从历史记录中查找最近的目标语言
        if st.session_state.translation_history:
            # 查找最近一次翻译的目标语言
            for item in reversed(st.session_state.translation_history):
                if "langs" in item and item["langs"]:
                    return item["langs"], True
    
    return [], False


def detect_document_reference(text):
    """
    检测用户输入中是否包含对之前文档的引用
    如"上面的文档"、"这个文件"等
    """
    # 文档引用关键词
    doc_keywords = [
        "上面的文档", "上面的文件", "上面的csv", "上面的数据",
        "这个文档", "这个文件", "这个csv", "这份文件", "这份文档",
        "之前的文档", "之前的文件", "之前的csv", "之前的数据",
        "刚才的文档", "刚才的文件", "刚才的csv",
        "那个文档", "那个文件", "那个csv",
        "上传的文档", "上传的文件", "上传的csv",
        "再翻译", "继续翻译", "重新翻译"
    ]
    
    text_lower = text.lower()
    is_doc_reference = any(k in text_lower for k in doc_keywords)
    
    return is_doc_reference


def find_text_column(df, api_key):
    """
    使用大模型智能判断哪列需要翻译
    """
    # 如果只有一列，直接返回
    if len(df.columns) == 1:
        return df.columns[0]
    
    # 准备列信息
    columns_info = []
    for col in df.columns:
        samples = df[col].dropna().head(3).tolist()
        samples_str = "\n  ".join([str(s)[:50] + ("..." if len(str(s)) > 50 else "") for s in samples])
        columns_info.append(f"列名: {col}\n   样本:\n   {samples_str}")
    
    columns_text = "\n\n".join(columns_info)
    
    prompt = f"""分析CSV列，选择需要翻译的文本列：

{columns_text}

返回JSON：{{"target_column": "列名"}}

标准：优先中文/日文/韩文文本，避免数字/日期/ID"""
    
    try:
        openai.api_key = api_key
        openai.api_base = "https://api.deepseek.com"
        
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "数据分析专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        
        try:
            result = json.loads(content)
            target_col = result.get("target_column")
            if target_col in df.columns:
                return target_col
        except:
            pass
        
        import re
        match = re.search(r'"target_column"\s*:\s*"([^"]+)"', content)
        if match:
            target_col = match.group(1)
            if target_col in df.columns:
                return target_col
        
    except:
        pass
    
    # 兜底：关键词匹配
    for col in df.columns:
        if any(k in str(col).lower() for k in ["中文", "内容", "文本", "原文", "text"]):
            return col
    
    return df.columns[0]


def get_data_source(uploaded_file, use_history=False, api_key=None):
    """
    获取数据源：优先使用上传的文件，如果没有则从历史记录获取
    """
    # 如果有新上传的文件，优先使用
    if uploaded_file is not None and not use_history:
        df = read_csv_with_encoding(uploaded_file)
        col = find_text_column(df, api_key)
        return df, col, "新上传的文件"
    
    # 尝试从历史记录获取原始数据
    if st.session_state.translation_history:
        last_item = st.session_state.translation_history[-1]
        if "source_df" in last_item:
            df = last_item["source_df"].copy()
            col = last_item.get("text_col", find_text_column(df, api_key))
            return df, col, "历史记录中的文档"
    
    return None, None, None


def translate_batch(texts, target_lang):
    prompt = f"""将以下文本翻译成{target_lang}，按JSON返回：
{{"translations": ["翻译1", "翻译2", ...]}}

文本：{json.dumps(texts, ensure_ascii=False)}"""
    
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"你是{target_lang}翻译专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
            if "translations" in result:
                return result["translations"][:len(texts)]
        except:
            pass
        
        # 备用解析
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if "translations" in result:
                    return result["translations"][:len(texts)]
            except:
                pass
        
        return texts
    except Exception as e:
        st.error(f"翻译出错：{e}")
        return texts


def process_translation(df, text_col, langs, api_key):
    openai.api_key = api_key
    openai.api_base = "https://api.deepseek.com"
    
    result = df.copy()
    texts = df[text_col].astype(str).tolist()
    batch_size = 20
    total = len(texts)
    
    progress_placeholder = st.empty()
    
    for lang in langs:
        with progress_placeholder.container():
            st.write(f"🔄 翻译 {lang}...")
            bar = st.progress(0)
        
        translations = []
        batches = (total + batch_size - 1) // batch_size
        
        for i in range(batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total)
            batch = texts[start:end]
            
            translations.extend(translate_batch(batch, lang))
            bar.progress((i + 1) / batches)
        
        result[f"{lang}_翻译"] = translations
    
    progress_placeholder.empty()
    return result


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI 翻译",
    page_icon="🌐",
    layout="centered"
)

# 自定义样式
st.markdown("""
<style>
    .block-container { max-width: 700px; padding: 2rem 1rem; }
    .stChatMessage { padding: 0.3rem 0; }
    
    /* 用户消息靠右 */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][data-testid="stChatMessageContent-user"]) {
        flex-direction: row-reverse !important;
    }
    .stChatMessage [data-testid="stChatMessageContent"][data-testid="stChatMessageContent-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        margin-right: 0;
    }
    
    /* 助手消息靠左 */
    .stChatMessage [data-testid="stChatMessageContent"][data-testid="stChatMessageContent-assistant"] {
        background: #f0f2f6;
        border-radius: 18px 18px 18px 4px;
    }
    
    div[data-testid="stFileUploader"] { border: 2px dashed #ddd; border-radius: 8px; padding: 1rem; }
    .stDownloadButton button { width: 100%; background: #4CAF50; color: white; border: none; padding: 0.8rem; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 📤 上传文件")
    uploaded_file = st.file_uploader("CSV 文件", type=["csv"])
    
    if uploaded_file:
        try:
            df_preview = read_csv_with_encoding(uploaded_file)
            st.success(f"✓ {len(df_preview)} 行数据")
        except Exception as e:
            st.error(f"读取失败：{e}")


# ==================== 主界面 ====================
st.markdown("<h1 style='text-align:center;'>🌐 AI 翻译 Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>上传 CSV，输入目标语言，一键批量翻译</p>", unsafe_allow_html=True)

# DeepSeek API Key (内置)
api_key = "sk-e4da75a05686471fa2c163db8314751f"


# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！上传 CSV 后告诉我目标语言，例如：*翻译成英语和日语*"}
    ]
if "last_langs" not in st.session_state:
    st.session_state.last_langs = []
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []


# 渲染单个历史记录的函数
def render_history_item(idx, item):
    with st.expander(f"📊 结果预览 - {', '.join(item['langs'])} ({len(item['result'])} 条)", expanded=True):
        st.dataframe(item['result'].head(5), use_container_width=True)
        
        csv = item['result'].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=f"⬇️ 下载翻译结果",
            data=csv,
            file_name=f"translated_{idx+1}_{'_'.join(item['langs'])}.csv",
            mime="text/csv",
            key=f"history_download_{idx}_{item['timestamp']}"
        )


# 显示消息 - 用户靠右，助手靠左
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 如果是助手消息且有对应的历史记录，显示预览和下载
        if msg["role"] == "assistant" and "history_idx" in msg:
            history_idx = msg["history_idx"]
            if history_idx < len(st.session_state.translation_history):
                item = st.session_state.translation_history[history_idx]
                render_history_item(history_idx, item)


# 聊天输入
if user_input := st.chat_input("输入翻译需求..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        # 检测是否包含文档引用
        is_doc_ref = detect_document_reference(user_input)
        
        # 检查是否有上传文件或历史记录可用
        has_uploaded_file = uploaded_file is not None
        has_history = bool(st.session_state.translation_history)
        
        if not has_uploaded_file and not has_history:
            st.warning("⚠️ 请上传 CSV 文件")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ 请上传 CSV 文件"})
        else:
            # 使用数据源（上传的文件或历史记录）
            use_history = is_doc_ref and not has_uploaded_file and has_history
            df, col, source_info = get_data_source(uploaded_file, use_history=use_history, api_key=api_key)
            
            if df is None:
                st.error("❌ 无法获取数据源")
                st.session_state.messages.append({"role": "assistant", "content": "❌ 无法获取数据源"})
            else:
                langs, is_context = detect_languages(user_input)
                
                if not langs:
                    st.info("🤔 我只能帮你进行翻译哦，快告诉我你需要翻译成什么语言吧")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "🤔 我只能帮你进行翻译哦，快告诉我你需要翻译成什么语言吧"
                    })
                else:
                    # 显示数据源信息
                    if use_history:
                        st.info(f"📂 自动使用{source_info}")
                    
                    # 如果是上下文引用，显示提示
                    if is_context:
                        st.info(f"💡 使用你之前提到的语言：{', '.join(langs)}")
                    else:
                        st.success(f"🎯 {', '.join(langs)}")
                    
                    # 保存当前语言到 session_state
                    st.session_state.last_langs = langs
                    
                    try:
                        st.caption(f"📄 {col} | {len(df)} 行")
                        
                        with st.expander("预览"):
                            st.dataframe(df[[col]].head(3), use_container_width=True)
                        
                        with st.spinner("翻译中..."):
                            result = process_translation(df, col, langs, api_key)
                        
                        # 保存到历史记录，包含原始数据
                        history_item = {
                            "langs": langs,
                            "result": result,
                            "source_df": df,  # 保存原始数据以便复用
                            "text_col": col,
                            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                        }
                        st.session_state.translation_history.append(history_item)
                        
                        st.success(f"✅ 完成！{len(df)} 条 → {len(langs)} 种语言")
                        
                        # 将结果预览和下载嵌入到当前助手消息中
                        with st.expander("📊 结果预览", expanded=True):
                            st.dataframe(result.head(5), use_container_width=True)
                            
                            csv = result.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="⬇️ 下载翻译结果",
                                data=csv,
                                file_name=f"translated_{len(st.session_state.translation_history)}_{'_'.join(langs)}.csv",
                                mime="text/csv",
                                key=f"download_{history_item['timestamp']}"
                            )
                        
                        # 记录此消息对应的历史索引
                        history_idx = len(st.session_state.translation_history) - 1
                        response_content = f"✅ 完成！已将 {col} 翻译成 {', '.join(langs)}，共 {len(df)} 条数据。"
                        
                    except Exception as e:
                        st.error(f"错误：{e}")
                        response_content = f"❌ 处理过程中出现错误：{e}"
                    
                    # 将助手回复添加到消息历史
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_content if 'response_content' in locals() else "处理完成",
                        "history_idx": history_idx if 'history_idx' in locals() else None
                    })
        
        # 重新运行以显示新消息
        st.rerun()
