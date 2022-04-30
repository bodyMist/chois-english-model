from flask import Flask, jsonify, request
from flask_restx import Api, Resource
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#Flask 객체 인스턴스 생성
app = Flask(__name__)
api = Api(app)
CORS(app)
model = SentenceTransformer('bert-base-nli-mean-tokens')


# 이미지 캡셔닝 요청 API
# @app.route('/', methods=['POST'])
# def caption():

#   return

# 단어 빈칸 채점 API
# POST Body : { user_input : 사용자 입력답안,  answer : 정답 , blank : 출제된 문제 빈칸,}
# Response : answers의 개수만큼 유사도 측정치 반환
@api.route('/score/<method>')
class wordScoring(Resource):
  def post(self, method):
    try:
      request_body = request.get_json()
      user_input = request_body['user_input']
      answer = request_body['answer']
      blank = request_body['blank']

      raw_similarity = self.compareWord(user_input, answer, blank) if method == "word" else self.compareSentence(user_input, answer, blank)[0]
      similarity = list(map(str, raw_similarity))
      index_name = ['word_similarity', 'sentence_similarity']

      response = dict(zip(index_name,similarity))

      return jsonify(str(response))
    except Exception as e:
      print(e)

  # 단어 채점 basic operation
  # user_sentence/answer & user_input/blank embedding 하고 비교
  def compareWord(self, user_input, answer, blank):
    print('\nStart Word Scoring')
    user_sentence = answer.replace(blank, user_input)
    # compare replaced sentence
    compareSentence = [user_sentence, answer]
    sentencesEmbedding = model.encode(compareSentence)
    sentencesEmbedding.shape
    sentenceCosine = (cosine_similarity(
        [sentencesEmbedding[0]],
        sentencesEmbedding[1:]
      )[0]
    )
    # compare raw word
    compareWord = [user_input, blank]
    wordEmbedding = model.encode(compareWord)
    wordEmbedding.shape
    wordCosine = cosine_similarity(
        [wordEmbedding[0]],
        wordEmbedding[1:]
      )[0]
          
    res = [wordCosine[0] , sentenceCosine[0]]
    return res

  # 문장 채점 basic operation
  # user_input/answer 비교
  # 이후 formatting 이슈 때문에 blank(answer)도 함께 비교
  def compareSentence(self, user_input, answer, blank):
    print('\nStart Sentence Scoring')
    compareSentence = [user_input, answer, blank]
    sentencesEmbedding = model.encode(compareSentence)
    
    sentencesEmbedding.shape
    res = cosine_similarity(
        [sentencesEmbedding[0]],
        sentencesEmbedding[1:]
      )
    return res


if __name__ == '__main__':
  # 코드 수정시 자동 반영
  app.run(host = '0.0.0.0', port=8000, debug=True)