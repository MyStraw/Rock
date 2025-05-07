import random # 랜덤함수
import pandas as pd #데이터프레임
import numpy as np #수치계산
import os #파일이나 폴더 경로 다룰때
import re # 정규표현식
import glob #특정 경로의 파일들 리스트로 가져오기
import cv2 #OpenCV: 이미지 불러오기 및 전처리
import timm #파이토치 기반 최신 이미지 모델 모음(ResNet등)

import torch #파이토치
import torch.nn as nn #모델 구성(레이어, 손실함수)
import torch.optim as optim #옵티마이저(아담)
import torch.nn.functional as F #활성화 함수, Loss함수수
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
#사용자 정의 데이터셋 상속, 배치단위 데이터 불러오기, 클래스 불균형시 오버샘플링 사용
# 아래는 이미지 전처리
import albumentations as A #강력한 이미지 증강 라이브러리
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import torchvision.models as models #기본 cnn모델들, ResNet등

from sklearn.model_selection import train_test_split #훈련/검증 나누기
from sklearn import preprocessing #라벨 인코딩
from sklearn.metrics import f1_score #성능 평가
from sklearn.metrics import classification_report #성능 평가
from tqdm.auto import tqdm #루프의 진행 상황을 한눈에 보기

import warnings #경고 메세지 안보이게
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 현재 gpu가 활성화 됐는지 확인
# false 일경우 cpu쓰기3
print(torch.cuda.is_available())      # True여야 정상
print(torch.cuda.get_device_name())  

CFG = { #딕셔너리, 하이퍼파리미터를 모아놓음. config 설정값
    #'IMG_SIZE':299, #299x299 리사이즈
    'IMG_SIZE':224, #resnet50 사이즈
    'EPOCHS':2, #학습 데이터셋 몇번 반복해서 학습할지
    'LEARNING_RATE':3e-4,# 0.0003 모델 가중치 조절. 학습률
    'BATCH_SIZE':16, #한번에 처리할 데이터 수. 원래 32
    'SEED':41 #난수 생성 시드값
}

def seed_everything(seed):
    random.seed(seed) #파이썬 내장 난수
    os.environ['PYTHONHASHSEED'] = str(seed) #해시 기반 연산-시드 영향 받게 환경변수 고정
    np.random.seed(seed) #넘파이 기반 난수 함수
    torch.manual_seed(seed) #파이토치에서 cpu 텐서 연산시 고정
    torch.cuda.manual_seed(seed) #파이토치에서 gpu 연산시 고정
    torch.backends.cudnn.deterministic = True #cnn 연산시 완전 동일한 연산 패턴 보장
    torch.backends.cudnn.benchmark = True #입력 사이즈 고정시 연산속도 최적화 가능(위와 충돌할수)

seed_everything(CFG['SEED']) # Seed 고정
#이 셀로 인해 같은 코드 + 같은 데이터로 같은 결과를 얻게 만드는 목적
#baseline을 기준으로 성능 비교 실험을 공정하게 하기위해.
#운빨요소 제거

all_img_list = glob.glob('./train/*/*')
#glob : 경로기반의 파일 리스트를 문자열로 가져오는 도구
#*/*는 두단계 하위 경로까지 검색색

print(len(all_img_list))
print(random.sample(all_img_list, 5))

df = pd.DataFrame(columns=['img_path', 'rock_type']) #df생성. 경로랑 락 타입 지정
df['img_path'] = all_img_list
#df['rock_type'] = df['img_path'].apply(lambda x : str(x).split('/')[2])
# 경로포맷 "\\" 로 저장될수도 있다. "/" 이거안먹히면 아래꺼꺼

from pathlib import Path
df['rock_type'] = df['img_path'].apply(lambda x: Path(x).parts[-2])

print(df)

train, val, _, _ = train_test_split( #_, _ 이건 test split 추가여지 남긴것
    df, 
    df['rock_type'], 
    test_size=0.2, #전체의 30%를 validation으로
    stratify=df['rock_type'], # 클래스 비율을 유지한채 분리
    random_state=CFG['SEED']) # 재현 가능한 split위해 시드고정
#train_test_split()함수는 기본적으로 : (X_train, X_test, y_train, y_test형태로 4개 반환)

le = preprocessing.LabelEncoder() # 문자열 라벨을 숫자로 바꿈
train['rock_type'] = le.fit_transform(train['rock_type']) #처음엔 학습 + 변환
val['rock_type'] = le.transform(val['rock_type']) #이미 학습된 매핑 그대로 숫자 변환

class PadSquare(ImageOnlyTransform): #이미지에 패딩 추가. 이미지별로 사이즈가 다르니 비율유지. 패딩으로 정사각형 만들고 resize함. ImageOnlyTransform 상속속
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w, c = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value) #이미지 테두리 확장함수수
        return image

    def get_transform_init_args_names(self):
        return ("border_mode", "value")
    
class CustomDataset(Dataset):
        
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index): #이미지 가져와서 BGR을 RGB로 전환환
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path) # OpenCV는 BGR 형식이다.
        
        if self.transforms is not None: #만약 transform이 주어지면 이미지 전처리 끝내기기
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None: #라벨 있으면 train, val 상황. 없으면 test 상황. 이미지만 만환환
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self): #데이터셋 크기 반환환
        return len(self.img_path_list)
    
train_transform = A.Compose([
    PadSquare(value=(0, 0, 0)), # 이미지 가로 세로중 긴쪽으로 패딩추가해 정사각형 맞추기기
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']), # 리사이즈. 299사이즈네 현재.
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), #이미지 정규화
    ToTensorV2() #파이토치 텐서로 전환환
])

test_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_dataset = CustomDataset(
    train['img_path'].values, 
    train['rock_type'].values, 
    train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size = CFG['BATCH_SIZE'], 
    shuffle=False, 
    num_workers=0, #원래 4 gpu 쓸땐 아래 3개 켜두기
    pin_memory=False, #원래 True
    #prefetch_factor=2)
)
# 한번에 학습시킬 이미지수(배치사이즈), 셔플은 하지않음, 트레인에선 True를 해야 오버피팅 막는데 좋다.
val_dataset = CustomDataset(
    val['img_path'].values, 
    val['rock_type'].values, 
    test_transform)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CFG['BATCH_SIZE'], 
    shuffle=False, 
    num_workers=0, #원래4 gpu 쓸땐 아래 3개 켜두기
    pin_memory=False, #원래 True
    #prefetch_factor=2
)

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device) #다중클래스 분류에서 쓰이는 손실함수.
    
    best_score = 0 #F1 스코어 기준 최고모델 저장
    best_model = None
    save_path = "best_model.pth"

    for epoch in range(1, CFG['EPOCHS'] + 1): #에퐄 1부터 반복
        model.train() #학습 시작
        train_loss = []

        for imgs, labels in tqdm(iter(train_loader), desc=f"Epoch {epoch}"): #tqdm 진행상황 시각화
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad() #기존 그래디언트 초기화
            output = model(imgs) #예측값 생성
            loss = criterion(output, labels) #예측값, 실제값 비교. 오차.

            loss.backward() #오차 역전파. 그래디언트 계산
            optimizer.step() #가중치 갱신(SGD나 아담 등등)

            train_loss.append(loss.item()) #로스 숫자만큼 추출해 리스트 저장장

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss: {_train_loss:.5f}, Val Loss: {_val_loss:.5f}, Val Macro F1: {_val_score:.5f}')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model

            # 모델 가중치 저장
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {save_path}")

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')
    
    return _val_loss, _val_score

#model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=7)
model = timm.create_model(
    'resnet18', 
    pretrained=True, 
    num_classes=7)

optimizer = torch.optim.Adam(
    params = model.parameters(), 
    lr = CFG["LEARNING_RATE"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=2, 
    threshold_mode='abs', 
    min_lr=1e-8, verbose=True) #학습률 줄어드는지 확인하기

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds

preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['rock_type'] = preds

submit.to_csv('./baseline_submit.csv', index=False)