import base64
from urllib import response
from flask import Flask, jsonify, request
from flask_restx import Api, Resource
from flask_cors import CORS
import json 

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#Flask 객체 인스턴스 생성
app = Flask(__name__)
api = Api(app)
CORS(app)
model = SentenceTransformer('bert-base-nli-mean-tokens')

# 단어 빈칸 채점 API
# POST Body : { user_input : 사용자 입력답안,  answer : 정답 , blank : 출제된 문제 빈칸,}
# Response : answers의 개수만큼 유사도 측정치 반환
@api.route('/score/<method>')
class WordScoring(Resource):
  def post(self, method):
    try:
      request_body = request.get_json()
      user_input = request_body['user_input']
      answer = request_body['answer']
      blank = request_body['blank']
      print("\nmethod : " ,method)
      print("user_input : " , user_input)
      print("answer : " , answer)
      print("blank : " , blank)
      raw_similarity = self.compareWord(user_input, answer, blank) if method == "word" else self.compareSentence(user_input, answer, blank)[0]
      similarity = list(map(str, raw_similarity))
      index_name = ['word_similarity', 'sentence_similarity']

      response = dict(zip(index_name,similarity))
      print(response)
      return jsonify(response)
    except Exception as e:
      print(e)

  # 단어 채점 basic operation
  # user_sentence/answer & user_input/blank embedding 하고 비교
  def compareWord(self, user_input, answer, blank):
    print('\nStart Word Scoring')
    # compare raw word
    compareWord = [user_input, blank]
    wordEmbedding = model.encode(compareWord)
    wordEmbedding.shape
    wordCosine = cosine_similarity(
        [wordEmbedding[0]],
        wordEmbedding[1:]
      )[0]
    
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


import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFile
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel

tasks.register_task('caption', CaptionTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# Load pretrained ckpt & config
overrides = {"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths('checkpoints/caption.pt'),
        arg_overrides = overrides
    )

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()

def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def image_caption(Image):
    sample = construct_sample(Image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)
    print(result)
    return result[0]['caption']

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import random

# delete stop_words from caption
# tokenize caption to create blank
def tokenizeCaption(caption):
  stop_words = set(stopwords.words('english')) 
  word_tokens = word_tokenize(caption)

  result = []
  for word in word_tokens: 
      if word not in stop_words: 
          result.append(word)
  blank = random.choice(result)
  return blank


import os
app.config['UPLOAD_FOLDER'] = './'

# 이미지 캡셔닝 요청 API
@api.route('/caption', methods=['POST'])
class Caption(Resource):
  def post(self):
    try:
      imageFile = request.files['file']
      print(type(imageFile))
      print(imageFile.headers)

      imageFile.save(imageFile.filename)
      image = Image.open(imageFile)
      image.show()

      # caption = image_caption(image)
      caption = "a girl with a bouquet of flowers"
      blank = tokenizeCaption(caption)
      # response = dict(zip("caption", caption))
      # return jsonify(response)
      response = {
        'success':True, 
        'caption': caption, 
        'blank':blank
      }
      os.remove(imageFile.filename)
      return jsonify(response)
    except Exception as e:
      print(e)

if __name__ == '__main__':
  # 코드 수정시 자동 반영
  app.run(host = '0.0.0.0', port=8000, debug=True)