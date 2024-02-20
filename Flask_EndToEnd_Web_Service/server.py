from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import transformers
import emoji
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
import random
random.seed(2024)
import numpy as np
import pandas as pd
import gc
import warnings
import urllib.request
from tqdm import tqdm
from glob import glob
from datetime import datetime, timedelta
import time
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from soynlp.normalizer import repeat_normalize
import gensim
from gensim import corpora
from konlpy.tag import Okt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# 파일 업로드 처리
@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        global review # 전역변수로 설정

        f = request.files['file']
        # 저장할 경로 + 파일명
        fixed_path = 'model_connect/static'
        upload_path = os.path.join(fixed_path, secure_filename('review_analysis.csv'))
        f.save(upload_path)
        review=pd.read_csv('model_connect/static/review_analysis.csv', encoding='utf-8')
        review = review[review['Type'] == 'Order'].reset_index(drop = True)
        review.rename(columns={'Content':'document'}, inplace=True)
        review = review[['ID', 'Created At', 'Rating', 'document', 'Order ID', 'User ID']]
        return render_template('index_afterupload.html')
    else : 
        return render_template('index.html')

class CustomDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, train_mode=True):
        self.dataset = dataset
        self.max_len = max_len
        self.train_mode = train_mode
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, 'document']
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            add_special_tokens=True
            )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        if self.train_mode:
            label = self.dataset.loc[idx, 'label']
            return input_ids, attention_mask, label
        return input_ids, attention_mask

    def __len__(self):
        return len(self.dataset)

class Report:

    def __init__(self, review, year, range):
        self.review = review
        self.year = year
        self.range = range
        self.is_week = (True if len(range[0]) == 7 else False)
        self.MODEL_NAME = {'electra':'kykim/electra-kor-base', 'funnel':'kykim/funnel-kor-base'}
        self.SAVE_PATH = './weight'
        self.NUM_CLASSES = 2
        self.BATCH_SIZE = 64
        self.EPOCHS = 10
        self.MAX_LEN = 50
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu') # Mac
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Windows

    def clean_text(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def text_preprocessing(self):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        self.review = self.review.dropna(subset = 'document').reset_index(drop = True)
        self.review['document'] = self.review['document'].apply(lambda x : self.clean_text(x))

    def predict(self, model_type, dataloader, weight_save_path, is_prob=False) -> np.array:
        """저장된 모델의 가중치를 불러와서 dataloader의 각 데이터를 예측하여 반환"""
        model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME[model_type], num_labels=self.NUM_CLASSES).to(self.device)
        weight_path_list = ['model_connect/model/electra_best_4.pt']
        # weight_path_list = glob(weight_save_path + '/*.pt')
        test_probs = np.zeros(shape=(len(dataloader.dataset), 2))

        for weight in tqdm(weight_path_list):
            if model_type not in weight:
                continue

            model.load_state_dict(torch.load(weight, map_location=self.device), strict = False)
            model.eval()
            probs = None

            with torch.no_grad():
                for input_ids, attention_masks in dataloader:
                    input_ids = input_ids.to(self.device)
                    attention_masks = attention_masks.to(self.device)

                    outputs = model(input_ids, attention_masks)[0]
                    outputs = outputs.cpu().numpy()

                    if probs is None:
                        probs = outputs
                    else:
                        probs = np.concatenate([probs, outputs])

            test_probs += (probs / 1)
        _, test_preds = torch.max(torch.tensor(test_probs), dim=1) ## 최대값과 인덱스

        if is_prob:
            return test_probs ## 각 컬럼별 확률
        return test_preds ## 라벨

    def sentiment_prediction(self):
        gc.collect()

        model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME['electra'], num_labels=self.NUM_CLASSES).to(self.device)
        electra_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME['electra'])

        test_dataset = CustomDataset(self.review, electra_tokenizer, self.MAX_LEN, train_mode = False)
        test_dataloader = DataLoader(test_dataset, batch_size = self.BATCH_SIZE, shuffle = False)

        preds = self.predict('electra', test_dataloader, self.SAVE_PATH, is_prob = True)

        predict_proba = self.softmax(preds)
        positive_ = predict_proba[:, 1]
        self.review['modified_score'] = positive_ * 100

    def softmax(self, x):
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp_x

    def report_features(self):
        self.review['Created At'] = self.review['Created At'].apply(lambda x : str(x)[:10])
        self.review['Created At'] = pd.to_datetime(self.review['Created At'], format = '%Y-%m-%d')
        self.review['year'] = self.review['Created At'].dt.year
        self.review['week'] = self.review['Created At'].dt.isocalendar().week
        self.review['month'] = self.review['Created At'].dt.month
        self.review['day'] = self.review['Created At'].dt.day
        
        self.review['cumsum_week'] = 0
        self.review['cumsum_day'] = 0

        self.review['cumsum_week'] = (self.review['year'] - 2021) * 52 + self.review['week'] + 3
        self.review.loc[(self.review['year'] == 2020) & (self.review['week'] == 46), 'cumsum_week'] = 1
        
        minimum_day = np.min(self.review['Created At'])
        self.review['cumsum_day'] = self.review['Created At'].apply(lambda x : (x - minimum_day).days + 1)
        self.review['modified_score'] = self.review['modified_score'].apply(lambda x : round(x, 2))
        self.review['sentiment_label'] = self.review['modified_score'].apply(lambda x : '긍정' if x >= 50 else '부정')

    def get_report_by_week(self):
        max = (int(self.year) - 2021) * 52 + int(self.range[1][5:]) + 3
        min = (int(self.year) - 2021) * 52 + int(self.range[0][5:]) + 3
        interval = max - min + 1

        min_week, max_week = self.get_minmax_week(min, max)  
        if min_week is None:
            print('해당 기간의 리뷰가 존재하지 않습니다. 다시 시도해 주십시오.')
            return
        
        comparison_target = (self.review['cumsum_week'] >= min_week) & (self.review['cumsum_week'] <= max_week)
        comparison_preview = (self.review['cumsum_week'] >= min_week - interval) & (self.review['cumsum_week'] <= max_week - interval)

        filter_target = self.review[comparison_target]
        filter_preview = self.review[comparison_preview]
        target_count = len(filter_target)
        target_preview = len(filter_preview)
        change_count = target_count - target_preview

        target_score = round(np.mean(filter_target['modified_score']), 2)
        preview_score = round(np.mean(filter_preview['modified_score']), 2)
        change_score = round(target_score - preview_score, 2)
        
        mean_score, std_score = round(np.mean(self.review['modified_score']), 2),  round(np.std(self.review['modified_score']), 2)
        expected_z_score = (target_score - mean_score) / std_score
        if expected_z_score >= 0.07:
            z_score_message = '다른 시기보다 사용자들의 반응이 매우 긍정적으로 나타나는 기간입니다.'
        elif expected_z_score >= 0.01:
            z_score_message = '다른 시기보다 사용자들의 반응이 주로 긍정적인 기간입니다.'
        elif expected_z_score >= -0.06:
            z_score_message = '긍정적인 반응과 부정적인 반응이 혼재하는 기간입니다.'
        else:
            z_score_message = '다른 시기에 비해 부정적인 반응이 주로 나타납니다. 적절한 민원 대응이 필요합니다.'
        

        target_positive_ratio = round(len(filter_target[filter_target['sentiment_label'] == '긍정']) / len(filter_target), 4) * 100
        preview_positive_ratio = round(len(filter_preview[filter_preview['sentiment_label'] == '긍정']) / len(filter_preview), 4) * 100
        change_ratio = round(target_positive_ratio - preview_positive_ratio, 2)

        review_count_message = f'해당 기간의 리뷰 수는 {len(filter_target)}개이며, {interval}주 전 대비 {change_count if (change_count) > 0 else change_count *(-1)}개가 ' + ('많습니다.' if change_count > 0 else '적습니다')
        score_message = f'평균 감성 점수는 {target_score}점이며, {interval}주 전 대비 {change_score if (change_score) > 0 else change_score * (-1)}점이 ' + ('상승하였습니다' if change_score > 0 else '하락하였습니다.')
        ratio_message = f'긍정 리뷰의 비율은 {target_positive_ratio : .2f}%이며, {interval}주 전 대비 {change_ratio if (change_ratio) > 0 else change_ratio * (-1)}%만큼 ' + ('상승하였습니다' if change_ratio > 0 else '하락하였습니다.')
        rank_message = f'해당 시기는 평균적으로 {mean_score - target_score : .2f}점 만큼 차이가 납니다. ' + z_score_message 
        top_reviews = filter_target.nlargest(5, 'modified_score')[['document', 'modified_score']]
        lowest_reviews = filter_target.nsmallest(5, 'modified_score')[['document', 'modified_score']]
        print('================== 요청해주신 주차의 리뷰에 대한 분석 결과입니다. ====================\n')
        print(f'{self.year}년의 {self.range[0]}주차부터 {self.range[1]}주차까지의 분석 결과입니다.')
        print(review_count_message)
        print(score_message)
        print(ratio_message)
        print(rank_message)
        print(f'{self.range[0]}주차부터 {self.range[1]}주차까지의 토픽 모델링 결과입니다.')
        self.topic_score(filter_preview, filter_target)
        topic_score, change = self.topic_score(filter_preview, filter_target)
        self.get_plot(min, max, topic_score, change, True)
        return review_count_message, score_message, ratio_message, rank_message, top_reviews, lowest_reviews

    def get_report_by_day(self):
        min_date = datetime.strptime(self.range[0].strip(), '%Y-%m-%d')
        max_date = datetime.strptime(self.range[1].strip(), '%Y-%m-%d')
        
        min_date, max_date = self.get_minmax(min_date, max_date)  
        if min_date is None:
            print('해당 기간의 리뷰가 존재하지 않습니다. 다시 시도해 주십시오.')
            return
        min = self.review[self.review['Created At'] == min_date]['cumsum_day'].iloc[0]
        max = self.review[self.review['Created At'] == max_date]['cumsum_day'].iloc[0]

        interval = max - min

        comparison_target = (self.review['cumsum_day'] >= min) & (self.review['cumsum_day'] <= max)
        comparison_preview = (self.review['cumsum_day'] >= min - interval) & (self.review['cumsum_day'] <= max - interval)
        filter_target = self.review[comparison_target]
        filter_preview = self.review[comparison_preview]
        target_count = len(filter_target)
        target_preview = len(filter_preview)
        change_count = target_count - target_preview

        target_score = round(np.mean(filter_target['modified_score']), 2)
        preview_score = round(np.mean(filter_preview['modified_score']), 2)
        change_score = round(target_score - preview_score, 2)
  
        mean_score, std_score = round(np.mean(self.review['modified_score']), 2), round(np.std(self.review['modified_score']), 2)
        expected_z_score = (target_score - mean_score) / std_score
        if expected_z_score >= 0.07:
            z_score_message = '다른 시기보다 사용자들의 반응이 매우 긍정적으로 나타나는 기간입니다.'
        elif expected_z_score >= 0.01:
            z_score_message = '다른 시기보다 사용자들의 반응이 주로 긍정적인 기간입니다.'
        elif expected_z_score >= -0.06:
            z_score_message = '긍정적인 반응과 부정적인 반응이 혼재하는 기간입니다.'
        else:
            z_score_message = '다른 시기에 비해 부정적인 반응이 주로 나타납니다. 적절한 민원 대응이 필요합니다.'
            
        target_positive_ratio = round(len(filter_target[filter_target['sentiment_label'] == '긍정']) / len(filter_target), 4) * 100
        preview_positive_ratio = round(len(filter_preview[filter_preview['sentiment_label'] == '긍정']) / len(filter_preview), 4) * 100
        change_ratio = round(target_positive_ratio - preview_positive_ratio, 2)
        difference = target_score - mean_score

        review_count_message = f'해당 기간의 리뷰 수는 {len(filter_target)}개이며, {interval}일 전 대비 {change_count if (change_count) > 0 else change_count *(-1)}개가 ' + ('많습니다' if change_count > 0 else '적습니다')
        score_message = f'평균 감성 점수는 {target_score}점이며, {interval}일 전 대비 {change_score if (change_score) > 0 else change_score * (-1)}점이 ' + ('상승하였습니다' if change_score > 0 else '하락하였습니다.')
        ratio_message = f'긍정 리뷰의 비율은 {target_positive_ratio : .2f}%이며, {interval}일 전 대비 {change_ratio if (change_ratio) > 0 else change_ratio * (-1)}%만큼 ' + ('상승하였습니다' if change_ratio > 0 else '하락하였습니다.')
        rank_message = f'해당 시기는 평균적으로 {difference : .2f}점 만큼 차이가 납니다. ' + z_score_message 
        top_reviews = filter_target.nlargest(5, 'modified_score')[['document', 'modified_score']]
        lowest_reviews = filter_target.nsmallest(5, 'modified_score')[['document', 'modified_score']]

        print('================== 요청해주신 주차의 리뷰에 대한 분석 결과입니다. ====================\n')
        print(f'{self.year}년의 {self.range[0]}일부터 {self.range[1]}일까지의 분석 결과입니다.')
        print(review_count_message)
        print(score_message)
        print(ratio_message)
        print(rank_message)
        print(f'{self.range[0]}일부터 {self.range[1]}일까지의 토픽 모델링 결과입니다.')
        self.topic_score(filter_preview, filter_target)
        topic_score, change = self.topic_score(filter_preview, filter_target)
        self.get_plot(min, max, topic_score, change, False)
        return review_count_message, score_message, ratio_message, rank_message, top_reviews, lowest_reviews

    def topic_inference(self):
        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다', '것', '엔', '접', '있다', '첫', '앞', '더', '분', '법', '차', '생', '고', '어', '글리', '어스', '알','더', '때', '집', '뜻'
                     , '디', '안', '수', '볼', '점', '제', '끼', '줄', '못', '데', '더', '주', '습', '날', '만', '또', '인', '양', '살', '번', '주', '점', '해', '용', '전', '청', '말', '거', '중', '순', '맛', '요', '향', '밥', '해', '더', '글', '반', '두', '제', '게', '걸', '뭐', '안', '왜', '뭘', '거', '양', '뜨', '꽤', '손', '때',
                     '이번', '항상', '늘', '류', '외', '구', '나', '회', '덜', '상', '마침', '모두', '해먹', '함', '총', '개', '비', '아주', '곳'
                     ]
        okt = Okt()
        
        dictionary = corpora.Dictionary.load('model_connect/model/lda topic modeling (0107).model.id2word')
        ldamodel = gensim.models.LdaModel.load('model_connect/model/lda topic modeling (0107).model')
        tokenized_sentence = []
        tokenized_doc = []
        for sentence in tqdm(self.review['document']):
            tokenized = okt.pos(str(sentence))
            stopword_removed = [word[0] for word in tokenized if word[1] in ['Noun'] and word[0] not in stopwords]
            tokenized_sentence.append(' '.join(stopword_removed))
            tokenized_doc.append(stopword_removed)

        new_text_corpus = [dictionary.doc2bow(text) for text in tokenized_doc]

        topictable = self.make_topictable_per_doc(ldamodel, new_text_corpus)
        topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
        
        self.review = pd.concat([self.review, topictable], axis = 1)
    
    def make_topictable_per_doc(self, ldamodel, corpus):
        topic_table = pd.DataFrame(columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중'])

        # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
        for i, topic_list in tqdm(enumerate(ldamodel[corpus])):
            doc = topic_list[0] if ldamodel.per_word_topics else topic_list
            doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
            # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
            # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
            # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
            # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

            # 모든 문서에 대해서 각각 아래를 수행
            for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
                if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                    topic_table.loc[i] = [i, int(topic_num), round(prop_topic, 4), topic_list]
                    # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
                else:
                    break
        return(topic_table)

    def topic_score(self, last, current):
        topic_scores = []
        change = []
        for i in range(5):
            filter_current = current[current['가장 비중이 높은 토픽'] == i]
            filter_last = last[last['가장 비중이 높은 토픽'] == i]
            
            pos_current = len(filter_current[filter_current['sentiment_label'] == '긍정'])
            neg_current = len(filter_current) - pos_current
            
            pos_last = len(filter_last[filter_last['sentiment_label'] == '긍정'])
            neg_last = len(filter_last) - pos_last

            changed = round(((pos_current - neg_current * 13.5) / (pos_current + neg_current)) - ((pos_last - neg_last * 13.5) / (pos_last + neg_last)), 3)
            topic_scores.append(round((pos_current - neg_current * 13.5) / (pos_current + neg_current), 3))
            change.append(changed)
            
        return topic_scores, change
        
    def get_plot(self, min, max, topic_scores, change, is_week = True):
        plt.figure(figsize = (100, 100))
        plt.subplots_adjust(hspace=0.5)

        min = min - 5
        trend = self.calculate_trend(min, max, is_week)
        if is_week: 
            trend['x_axis'] = trend['year'] + '-' + trend['week']
            min_score, max_score = np.min(trend['modified_score']), np.max(trend['modified_score'])
            min_axis, max_axis = trend[trend['modified_score'] == min_score]['x_axis'], trend[trend['modified_score'] == max_score]['x_axis']
        else:
            trend['x_axis'] = trend['month'] + '-' + trend['day']
            min_score, max_score = np.min(trend['modified_score']), np.max(trend['modified_score'])
            min_axis, max_axis = trend[trend['modified_score'] == min_score]['x_axis'], trend[trend['modified_score'] == max_score]['x_axis']       
        
        x_label = ('Year & Week' if is_week else 'Day')
        title_name = 'Sentiment score trend of last ' + str((max-min)+1) + (' weeks ' if is_week else ' days ') + ('in ' + str(self.year) if is_week is not True else '')
        
        title_font = {'size': 120, 'family': 'Arial', 'weight':'bold'}
        subplot_font = {'size': 120, 'family': 'Arial'}

        plt.subplot(3, 1, 1)
        sns.lineplot(x = trend['x_axis'], y = trend['modified_score'], marker = 'o', color = 'black', linewidth=5)
        plt.xlabel(x_label, subplot_font)
        plt.xticks(fontsize=60)
        plt.ylabel('Sentiment score', subplot_font)
        plt.yticks(fontsize=60)
        plt.ylim([65, 100])
        plt.title(title_name, title_font)
        plt.axhline(y=87.04, color='b', linestyle='--', linewidth=5)
        plt.annotate('Lowest score\n' + ': ' + str(round(min_score,2)), xy = (min_axis, min_score), xytext = (min_axis, min_score - 1),
                    arrowprops = dict(facecolor = 'red', arrowstyle = '->, head_length = 0.5, head_width = 0.5'), color = 'red',
                    weight='bold', fontsize=80)
        plt.annotate('Highest score\n' + ': ' + str(round(max_score,2)), xy = (max_axis, max_score), xytext = (max_axis, max_score - 1),
                    arrowprops = dict(facecolor = 'red', arrowstyle = '->, head_length = 0.5, head_width = 0.5'), color = 'red',
                    weight='bold', fontsize=80)

        x = [0, 1, 2, 3, 4]
        min_topicscore, max_topicscore, min_changescore, max_changescore = np.min(np.array(topic_scores)), \
            np.max(np.array(topic_scores)), np.min(np.array(change)), np.max(np.array(change))
        min_index, max_index, min_changeindex, max_changeindex = np.argmin(np.array(topic_scores)), np.argmax(np.array(topic_scores)), \
            np.argmin(np.array(change)), np.argmax(np.array(change))
        
        plt.subplot(3, 1, 2)
        sns.barplot(x = x, y = topic_scores, color='#F78604')
        plt.xticks(ticks=x, labels=['포장 & 친환경', '주문 & 상태', '배송 & 이유식', '요리 & 레시피', '품목 구성'], fontsize=80, fontfamily='AppleGothic')
        plt.yticks(fontsize=60)
        plt.title('Topic score for each topic during the input period', title_font)
        plt.xlabel('Topic', subplot_font)
        plt.ylabel('Topic score', subplot_font)
        plt.axhline(y=-0.6, color='b', linestyle='--', linewidth=5)
        plt.axhline(0, color = 'black', linewidth = 0.6)
        fixed_y_position_ratio = 0.8
        fixed_y_position_max_a = max_topicscore * fixed_y_position_ratio
        fixed_y_position_min_a = min_topicscore * fixed_y_position_ratio

        plt.annotate('Lowest score : ' + str(round(min_topicscore,3)), xy = (min_index, min_topicscore), ha='center', va='center', 
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'), xytext = (min_index, fixed_y_position_min_a), color = 'black', weight='bold', fontsize=60)
        plt.annotate('Highest score : ' + str(round(max_topicscore,3)), xy = (max_index, max_topicscore), ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'), xytext = (max_index, fixed_y_position_max_a), color = 'black', weight='bold', fontsize=60)
                
        plt.subplot(3, 1, 3)
        sns.barplot(x = x, y = change, color = '#FEFD48')
        plt.xticks(ticks=x, labels=['포장 & 친환경', '주문 & 상태', '배송 & 이유식', '요리 & 레시피', '품목 구성'], fontsize=80, fontfamily='AppleGothic')
        plt.yticks(fontsize=60)
        plt.title('Same period as the search period and amount of change by topic', title_font)
        plt.xlabel('Topic', subplot_font)
        plt.ylabel('Changes compared to last period', subplot_font)
        plt.axhline(0, color = 'black', linewidth = 0.6)
        fixed_y_position_ratio = 0.8
        fixed_y_position_max_b = max_changescore * fixed_y_position_ratio
        fixed_y_position_min_b = min_changescore * fixed_y_position_ratio

        plt.annotate('Lowest changed score\n' + ': ' + str(round(min_changescore,3)), xy = (min_changeindex, min_changescore), 
                     ha='center', va='center', xytext = (min_changeindex, fixed_y_position_min_b), color = 'black', weight='bold', 
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'), fontsize=60)
        plt.annotate('Highest changed score\n' + ': ' + str(round(max_changescore,3)), xy = (max_changeindex, max_changescore), 
                     ha='center', va='center', xytext = (max_changeindex, fixed_y_position_max_b), color = 'black', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'), fontsize=60)
        plt.savefig('model_connect/static/output_plot.png')
        plt.close()

    def calculate_trend(self, min, max, is_week = True):
        if is_week:
            filter = self.review[(self.review['cumsum_week'] >= min) & (self.review['cumsum_week'] <= max)]
            sentiment_trend = filter.groupby('cumsum_week')[['year', 'week', 'modified_score']].agg('mean')
            sentiment_trend['year'] = sentiment_trend['year'].astype(int).astype(str)
            sentiment_trend['week'] = sentiment_trend['week'].astype(int).astype(str)
        else:
            filter = self.review[(self.review['cumsum_day'] >= min) & (self.review['cumsum_day'] <= max)]
            sentiment_trend = filter.groupby('cumsum_day')[['year', 'month', 'day', 'modified_score']].agg('mean')
            sentiment_trend['year'] = sentiment_trend['year'].astype(int).astype(str)
            sentiment_trend['month'] = sentiment_trend['month'].astype(int).astype(str)
            sentiment_trend['day'] = sentiment_trend['day'].astype(int).astype(str)
        return sentiment_trend.reset_index()
    
    def get_minmax(self, min_date, max_date):
        if (len(self.review[self.review['Created At'] == min_date]) != 0) & (len(self.review[self.review['Created At'] == max_date]) != 0):
            return min_date, max_date
        else:
            while min_date <= max_date:
                if len(self.review[self.review['Created At'] == min_date]) == 0:
                    min_date = min_date + timedelta(days = 1)
                    continue
                if len(self.review[self.review['Created At'] == max_date]) == 0:
                    max_date = max_date - timedelta(days = 1)
                    continue
                return min_date, max_date     
        if min_date > max_date:
            return None, None
            
    def get_minmax_week(self, min_week, max_week):
        if (len(self.review[self.review['cumsum_week'] == min_week]) != 0) & (len(self.review[self.review['cumsum_week'] == max_week]) != 0):
            return min_week, max_week
        else :
            while min_week <= max_week:
                if len(self.review[self.review['cumsum_week'] == min_week]) == 0:
                    min_week += 1
                    continue
                if len(self.review[self.review['cumsum_week'] == max_week]) == 0:
                    max_week -= 1
                    continue
                return min_week, max_week
        if min_week > max_week:
            return None, None
        return min_week, max_week

    def get_report_card(self):
        start = time.time()
        print('####################### 리뷰를 전처리하고 있습니다. #######################')
        self.text_preprocessing() # 전처리 과정
        print('####################### 리뷰 전처리가 완료되었습니다. #######################')
        print('####################### 감성 분석이 진행되고 있습니다. #######################')
        self.sentiment_prediction() # 예측 과정
        print('####################### 감성 분석을 완료하였습니다. #######################')
        end = time.time()
        print(end - start)
        print('####################### 요청하신 주차에 대한 리포트 내용을 준비중입니다. #######################')
        self.report_features()
        self.topic_inference()
        
        if self.is_week:
            return self.get_report_by_week()
        else:
            return self.get_report_by_day()

@app.route('/report', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        report_type = request.form.get('report_type')

        # 웹페이지 상 날짜 선택
        ## 일별 추출
        if report_type == 'daily' : 
            try : 
                start_date_str = request.form.get('start_date')
                end_date_str = request.form.get('end_date')
                year = start_date_str[:4]
                report = Report(review, year, [start_date_str, end_date_str])
                review_count_message, score_message, ratio_message, rank_message, top_reviews, lowest_reviews = report.get_report_card()
                image_path = 'model_connect/static/output_plot.png'
                return render_template('report.html', start_date = start_date_str, end_date = end_date_str, review_count_message=review_count_message,
                                    score_message=score_message, ratio_message=ratio_message, rank_message=rank_message, top_reviews=top_reviews,
                                    lowest_reviews=lowest_reviews, chart_image=image_path)
            except Exception as e : 
                print('error 내용 : ', e)
                return render_template('index.html')
        
        ## 주별 추출
        elif report_type == 'weekly' : 
            try : 
                start_year = request.form['start_year']
                start_week = request.form['start_week']
                end_year = request.form['end_year']
                end_week = request.form['end_week']
                start_week_submit = f"{start_year}-{start_week}"
                end_week_submit = f"{end_year}-{end_week}"
                year = start_year[:4]
                report = Report(review, year, [start_week_submit, end_week_submit])
                review_count_message, score_message, ratio_message, rank_message, top_reviews, lowest_reviews = report.get_report_card()
                image_path = 'model_connect/static/output_plot.png'
                return render_template('report.html', start_date = start_week_submit, end_date = end_week_submit, start_week = start_week_submit, end_week = end_week_submit, review_count_message=review_count_message,
                                   score_message=score_message, ratio_message=ratio_message, rank_message=rank_message, top_reviews=top_reviews,
                                   lowest_reviews=lowest_reviews, chart_image=image_path)
            except Exception as e : 
                print('error 내용 : ', e)
                return render_template('index.html')
            
    # GET 요청
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
