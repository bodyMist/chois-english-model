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

# 정답 채점 API
# POST Body : { user_input : 사용자 입력답안,  answer : 정답 , blank : 출제된 문제 빈칸,}
# Response : answers의 개수만큼 유사도 측정치 반환
# OFA에서 다수의 출력문이 나온다 ? 
# 유사도를 높이기 위해 해당 출력문을 모두 사용하는 방안이 좋겠지만,
# HTTP 특성상, stateless이기 때문에, 이 방안을 사용하려면 captioning 출력문을 전부 client에 전송해야함

# API가 통합되어 있으므로 로직도 통합 구현 => 단어/문장 모두 blank/user_input 비교 & answer/user_input 비교
@api.route('/score')
class scoring(Resource):
  def post(self):
    try:
      request_body = request.get_json()
      user_input = request_body['user_input']
      answer = request_body['answer']
      blank = request_body['blank']
      similarity = list(map(str, self.compareSimilarity(user_input, answer, blank)[0]))
      index_name = ['sentence_similarity','blank_similarity']

      response = dict(zip(index_name,similarity))

      return jsonify(str(response))
    except Exception as e:
      print(e)

  # basic operation
  # user_input/answer & user_input/blank 각각에 대해 embedding 하고 비교
  def compareSimilarity(self, user_input, answer, blank):
    sentences = [user_input, answer, blank]
    sentences_embeddings = model.encode(sentences)
    sentences_embeddings.shape
    res = cosine_similarity(
        [sentences_embeddings[0]],
        sentences_embeddings[1:]
      )
    return res




if __name__ == '__main__':
  # 코드 수정시 자동 반영
  app.run(host = '0.0.0.0', port=8000, debug=True)