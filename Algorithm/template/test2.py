import sqlite3
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jieba
import matplotlib.font_manager as fm
from pathlib import Path

# ========== 1. 读取聊天数据 ==========

db_path = Path(r"F:\下载\zwx\wechatDataBackup_wxid_r1rjahxfwnhl22\User\wxid_vhzdj3nflvoa22\Msg\Multi\MSG.db")
conn = sqlite3.connect(db_path)

msg_df = pd.read_sql_query("SELECT IsSender, CreateTime, StrTalker, StrContent, DisplayContent, Type FROM MSG", conn)
text_df = msg_df[msg_df['Type'] == 1].copy()

text_df['Time'] = text_df['CreateTime'].apply(lambda x: datetime.datetime.fromtimestamp(x))
text_df['Sender'] = text_df['IsSender'].apply(lambda x: '我' if x == 1 else '对方')
text_df['Content'] = text_df['StrContent'].fillna('')

output_df = text_df[['Time', 'Sender', 'Content']].sort_values(by='Time')
output_df.to_csv('chat_log.csv', index=False, encoding='utf-8-sig')
print("✅ 已导出 chat_log.csv")

# ========== 2. 统计分析 ==========

# 中文字体路径
zh_font = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")  # 微软雅黑路径

total_msgs = len(output_df)
total_chars = output_df['Content'].str.len().sum()
msg_count_by_sender = output_df['Sender'].value_counts()

output_df['Date'] = output_df['Time'].dt.date
daily_counts = output_df.groupby(['Date', 'Sender']).size().unstack(fill_value=0)

latest_msg = output_df.loc[output_df['Time'].idxmax()]

output_df['TimeDelta'] = output_df['Time'].diff().fillna(pd.Timedelta(seconds=0))
output_df['GapFlag'] = (output_df['TimeDelta'] > pd.Timedelta(minutes=10)).cumsum()
chat_sessions = output_df.groupby('GapFlag').agg({'Time': ['min', 'max', 'count']})
chat_sessions['Duration'] = chat_sessions[('Time', 'max')] - chat_sessions[('Time', 'min')]
longest_session = chat_sessions.sort_values(by=('Duration'), ascending=False).iloc[0]

print(f"\n📊 聊天统计：")
print(f"总消息数：{total_msgs}")
print(f"总字数：{total_chars}")
print(f"消息数（按人）：\n{msg_count_by_sender}")
print(f"最晚一条消息时间：{latest_msg['Time']}，内容：{latest_msg['Content']}")
print(f"最长一次连续聊天：{longest_session['Duration']}，共 {longest_session[('Time', 'count')]} 条消息")

# ========== 3. 词频分析 ==========

all_text = ''.join(output_df['Content'].tolist())
words = jieba.lcut(all_text)
stopwords = set(['的', '了', '我', '你', '在', '是', '就', '都', '和', '也', '不', '还', '有', '啊', '嘛', '吗', '哦', '吧', '呢', '呀', '啦'])
filtered_words = [w for w in words if len(w) > 1 and w not in stopwords]
word_freq = pd.Series(filtered_words).value_counts().head(20)
print("\n📝 最高频词：\n", word_freq)

# 条形图替代词云
plt.figure(figsize=(12, 6))
sns.barplot(x=word_freq.values, y=word_freq.index, palette="viridis")
plt.title("聊天中最常用的词汇（Top 20）", fontproperties=zh_font)
plt.xlabel("出现次数")
plt.ylabel("词语", fontproperties=zh_font)
plt.tight_layout()
plt.savefig("top_words_bar.png")
plt.show()

# 每日消息趋势图
daily_counts.plot(kind='line', figsize=(12, 6), title='每日消息趋势', marker='o')
plt.xlabel('日期')
plt.ylabel('消息数')
plt.xticks(rotation=45)
plt.title('每日消息趋势', fontproperties=zh_font)
plt.legend(prop=zh_font)
plt.tight_layout()
plt.savefig('daily_trend.png')
plt.show()
