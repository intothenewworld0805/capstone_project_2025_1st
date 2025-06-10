import pandas as pd

# 1. 엑셀 파일 불러오기
df = pd.read_excel('250604_snsd_data_labeling_2000_Into_The_New_World_0513_02_only_comments_english.xlsx')  # 예: 'labeled_comments.xlsx'

# 2. CSV 파일로 저장하기
df.to_csv('250604_snsd_data_labeling_2000_Into_The_New_World_0513_02_only_comments_english.csv', index=False, encoding='utf-8-sig')  # 예: 'labeled_comments.csv'
