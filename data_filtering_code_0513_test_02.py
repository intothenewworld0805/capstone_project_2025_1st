from langdetect import detect
import json
import re
from tqdm import tqdm  # tqdm을 추가합니다

INPUT_FILE = '../data_crawling_code/snsd_all_comments_Into_The_New_World_0513_01_only_comments.jsonl'
OUTPUT_FILE = 'snsd_all_comments_Into_The_New_World_0513_02_only_comments_english.jsonl'


def clean_text(text):
    """
    텍스트에서 특수 문자와 불필요한 공백 제거
    """
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)  # 특수 문자 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 여러 공백을 하나로
    cleaned_text = cleaned_text.strip()  # 앞뒤 공백 제거
    return cleaned_text


def filter_english_comments(input_file, output_file):
    """
    영어 댓글만 필터링하여 새 파일에 저장
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        comment_count = 0
        total_comments = sum(1 for line in infile)  # 전체 댓글 수를 구함
        infile.seek(0)  # 파일의 처음으로 돌아갑니다 (파일 포인터를 처음으로 리셋)

        # tqdm 진행 상황 표시
        with tqdm(total=total_comments, desc="필터링 중", unit="개") as pbar:
            for line in infile:
                try:
                    comment_data = json.loads(line)
                    comment_text = comment_data.get("text", "").strip()

                    # 텍스트가 비어 있거나 너무 짧으면 필터링
                    if len(comment_text) < 10:
                        pbar.update(1)  # 진행 상태 업데이트
                        continue

                    # 특수 문자 처리
                    cleaned_text = clean_text(comment_text)

                    # 언어 감지
                    language = detect(cleaned_text)
                    if language == 'en':  # 영어로 감지되면
                        json.dump(comment_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        comment_count += 1

                except Exception as e:
                    print(f"⚠️ 오류 발생: {e}")

                pbar.update(1)  # 진행 상태 업데이트

        print(f"✅ 영어 댓글 수집 완료: {comment_count}개")


# 💡 실행
if __name__ == "__main__":
    print("🔍 영어 댓글 필터링 시작...")
    filter_english_comments(INPUT_FILE, OUTPUT_FILE)
    print(f"💾 영어 댓글 저장 완료: {OUTPUT_FILE}")
