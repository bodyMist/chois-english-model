from typing import final
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
# POST Body : { user_input : 사용자 입력답안, blank : 출제된 문제 빈칸, ...answers : 정답 }
# Response : answers의 개수만큼 유사도 측정치 반환
@api.route('/score')
class scoring(Resource):
  def post(self):
    try:
      print(request.get_json())
      sentences = ["swimming pool", "water", "sea"]
      sentences_embeddings = model.encode(sentences)
      sentences_embeddings.shape

      res = cosine_similarity(
        [sentences_embeddings[0]],
        sentences_embeddings[1:]
      )
      print(res[0])
      # res [[0.67601085 0.6500472 ]]
      return jsonify("hi")
    except Exception as e:
      print(e)




if __name__ == '__main__':
  # 코드 수정시 자동 반영
  app.run(host = '0.0.0.0', port=8000, debug=True)