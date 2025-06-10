import json
import csv

# 입력 파일명과 출력 파일명 설정
input_file = 'snsd_all_comments_Into_The_New_World_0513_01_only_comments.jsonl'
output_file = 'snsd_all_comments_Into_The_New_World_0513_01_only_comments.csv'

# JSONL 파일을 CSV로 변환
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = None
    for line in infile:
        data = json.loads(line)

        # 첫 줄에서 CSV 헤더 작성
        if writer is None:
            writer = csv.DictWriter(outfile, fieldnames=data.keys())
            writer.writeheader()

        writer.writerow(data)

print("CSV 파일로 성공적으로 변환되었습니다.")
