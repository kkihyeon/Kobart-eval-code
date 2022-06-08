# Kobart-eval-code

제가 kobart 모델 관련하여 장난을 쳐본적이 있습니다. 근데 제가 학습시킨 모델을 어떻게 구체적인 코드로 평가할지에 대해서는 잘 모르겠더군요 :)
그래서 여러 코드를 찾아보다 kobart 평가 관련해서 eval관련 평가 방법이 있길래 이를 직접 수정하여 사용하였습니다.

에 그래서 rouge척도를 기반으로 kobart 자연어 처리 모델을 평가하는 코드를 설명드리려 합니다.
kobart모델에 대해 학습한 모델에 대하여 평가를 진행할 때 rouge_metric.py를 이용하시게 될 것 같습니다.
rouge_metric.py는 KoBART-summarization게시글을 참고하거나 SKT에서 올려주신 KoBART게시글을 참고하셔서 코드를 얻으시기 바랍니다.
그냥 제가 올려드리겠습니다.

rouge_metric.py내 소스 코드를 보면 전체적으로 Rouge클래스가 있고 클래스 내 여러 함수로 이루어져 있는 것을 확인할 수 있습니다.
소스코드의 Rouge클래스는 기본적으로 사용할 수 있는 metric으로 rouge-n, rouge-l, rouge-w가 있고, 클래스 객체를 선언하는 과정에서 metric을 선택하여 옵션을 넣어줄 수 있습니다.

ex)
<pre><code>
rouge = Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'])
#예시의 경우에는 모든 metric을 옵션으로 준 것입니다.
</code></pre>

객체를 선언하실 때 metric옵션을 주셨으면 이제 실제 평가를 해서 결과값을 도출해내는 함수를 이용하셔야 하는데, 제 생각에는 클래스 내 여러 함수 중 get_scores()가 중점적인 역할을 하는 것으로 보입니다. 해당 함수를 기점으로 클래스 내 타 함수가 서로 엮여서 동작하는 것 같더라구요.

get_scores()는 인자를 (인공지능의 요약 데이터, 라벨데이터) 이렇게 두 가지를 받고 내부적으로 tokenizing하여 앞서 준 metric 옵션대로 연산 후 평가 값을 도출해내는 함수인 것 같습니다.
ex)
<pre><code>
result = rouge.get_scores(summarize, label)
#summarize는 kobart 인공지능 모델의 요약 데이터(string)
#label은 비교의 기준이 될만한 라벨 데이터(string)
</code></pre>
get_scores()를 사용하여 평가하면 총 세가지 버전으로 평가 값을 얻을 수 있습니다. f1, precision, recall 이렇게요. 
보통 rouge척도에서 두 데이터 간 비교를 할 때 해당 세가지 연산이 주로 이용되는 것 같습니다. recall은 라벨데이터에 대해 요약데이터가 얼마나 겹치는지, precision은 요약데이터에 대해 라벨데이터가 얼마나 겹치는지, f1은 앞선 두 방법을 종합적으로 고려한 연산. 이렇게 일단은 파악했습니다. 그래서 recall과 precision은 어느 한 쪽에 편파적인 값이 나올 수 있어서 주로 f1이 중점적으로 활용되는 것으로 이해했습니다.

그래서 앞선 get_scores()의 결과값으로 {'rouge-l':  f:~~~, ,p:~~~, r:~~~} 이런식으로 결과값이 나오게 될 것입니다. 결과 타입은 딕셔너리입니다. value는 float형태이니 소수점을 적당히 자르셔서 쓰시기 바랍니다. 제가 get_scores()를 실행시킨 결과로 0.4xx나 0.5xx이렇게 나오더군요. 

지금까지 설명드린 코드는 한 문장에 대해서 입력하고 출력값은 얻는 것을 보인 겁니다. 여러분이 여러 문장을 평가하고 이를 평균내어 종합 결과를 얻고 싶으면 반복문으로 구성하셔서 이용하시면 됩니다.

혹시 몰라 제 코드도 같이 넣어놓겠습니다.(별로 좋은 코드는 아니어서 참고만 하시기 바랍니다^^)
<pre>
<code>
rouge = Rouge(metrics=['rouge-n','rouge-l','rouge-w'])
s_file = 'trainTT.json' #평가할 데이터셋(요약본과 라벨 둘 다 들어있음)
rouge_l = [0,0,0]   #각각 f,p,r
rouge_w = [0,0,0]   #각각 f,p,r

with open(s_file, 'r', encoding="UTF-8") as f:
    st_python = json.load(f)

#tqdm()은 단지 코드 수행 과정을 가시적으로 보기 위함이라 신경쓰지 않으셔도 됩니다.
for i in tqdm(range(len(st_python)),desc="(평가중...)",ascii=True):
    sumR = json.dumps(st_python[i][0],ensure_ascii=False)
    labelR = json.dumps(st_python[i][1],ensure_ascii=False)
    result = rouge.get_scores(sumR,labelR)

    rouge_l[0] += Tresult['rouge-l']['f']
    rouge_l[1] += Tresult['rouge-l']['p']
    rouge_l[2] += Tresult['rouge-l']['r']
    rouge_w[0] += Tresult['rouge-w']['f']
    rouge_w[1] += Tresult['rouge-w']['p']
    rouge_w[2] += Tresult['rouge-w']['r']

    for j in range(len(rouge_l)):
        rouge_l[j] /= (i+1)
        rouge_w[j] /= (i+1)

print(rouge_l, rouge_w)
</code>
</pre>

+추가적으로 원래 rouge_metric.py에서는 문장을 tokenizing 하는 과정에서  mecab라이브러리를 이용한 것 같습니다.
에...하지만 제 컴퓨터에서는 mecab 환경 구성이 잘 안되더군요ㅠ 그래서 이미 있는 kobart_tokenizer를 이용했습니다. 둘 다 한국어 문장을 토큰화 시켜주는 역할을 하는 것 같아서 대체해서 썼습니다. 일단은 필요하신 분들을 위해 원래 rouge_metric.py코드에서 어떤 부분을 수정했는지 적겠습니다.

일단은 모듈 import 부분을 보시면
<pre><code>
from konlpy.tag import Mecab
</code></pre>
이렇게 되어있으실 건데 저는 대신에 kobart tokenizer를 이용했기 때문에
<pre><code>
from kobart import get_kobart_tokenizer
</code></pre>
를 넣어주었습니다.

그리고 __init__에서 tokenizer를 선정하는 코드가 있습니다.
<pre><code>
self.use_tokenizer = use_tokenizer
if use_tokenizer:
    self.tokenizer = Mecab()
</code></pre>
이 부분에서 Mecab()을 get_kobart_tokenizer()로 수정했습니다.

또 수정할 부분이 하나 더 남았는데 문장을 받아 실제 토큰화 하는 함수입니다.
아까와 같은 __init__를 보시면 tokenize_text부분이 있는데,
<pre><code>
def tokenize_text(self, text):
    if self.use_tokenizer:
        return self.tokenizer.morphs(text)
    else:
        return text
</code></pre>
이 중 return self.tokenizer.morphs(text)를 수정해야하는데요 morphs()같은 경우 mecab에서 쓰는 함수로 이해했습니다. 해당 코드를
return self.tokenizer.tokenize(text)로 수정했습니다.

이렇게 수정하시면 mecab 대신에 kobart_tokenizer를 이용할 수 있으실 겁니다.

제가 혼자서 분석한 것이기 때문에 오개념이 포함될 가능성이 있습니다. 잘못된 부분이 있으면 지적 부탁드립니다. 감사합니다.
