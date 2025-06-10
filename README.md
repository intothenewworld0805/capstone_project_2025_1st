# MobileBERT를 활용한 소녀시대 다시 만난 세계 뮤직비디오 댓글 감성 분석 프로젝트
# 1. 개요
이 프로젝트는 K-pop을 대표하는 명곡 소녀시대의 ‘다시 만난 세계 (Into The New World)’ 뮤직비디오에 달린 영어 댓글을 수집하고, 이를 분석하여 사람들의 감정 반응을 MobileBERT 기반 감성 분석 모델로 파악하는 것을 목표로 합니다. 선정한 곡은 한국 대중문화에서 상징적인 의미를 가지며, 글로벌 K-pop 팬들 사이에서도 지속적으로 언급되는 곡입니다. 본 프로젝트는 전 세계 사람들이 해당 곡에 대해 어떤 감정을 표현하고 있는지를 데이터 기반으로 분석하고 시각화함으로써, 문화적·정서적 반응의 흐름을 이해하고자 합니다.
# 2. 데이터
### 📄 JSONL 원시 데이터 형식 예시

| 필드명             | 설명                                                                 |
|--------------------|----------------------------------------------------------------------|
| `author`           | 댓글 작성자의 닉네임                                                  |
| `author_channel_id`| 댓글 작성자의 YouTube 채널 ID                                         |
| `text`             | 댓글 내용 (본문)                                                      |
| `published_at`     | 댓글이 작성된 날짜 및 시간 (ISO 8601 형식)                             |
| `like_count`       | 해당 댓글이 받은 좋아요 수                                            |

#### 예시 (1개 댓글)

```json
{
  "author": "@2oqp577",
  "author_channel_id": "UCsCaKV9NGR7lwFxX0LZHB6w",
  "text": "What a fantastic time it was for the girls, k-pop and Korea. I am marked by this time when I got acquainted with Korea, it`s culture, language and history. Godspeed Korea!",
  "published_at": "2025-05-12T03:16:41Z",
  "like_count": 3
}
