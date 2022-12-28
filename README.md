# Token Classification with PLM

## Data

- [KLUE](https://github.com/KLUE-benchmark/KLUE)
- [한국해양대학교 NER 데이터셋](https://github.com/kmounlp/NER)
  - 본 데이터셋에서는 PER, LOC, ORG, DAT, TIM, DUR 만을 사용했습니다.
  - klue 및 한국해양대학교 NER 데이터셋은 klue 로 명명되었으며 호출 시 구분 없이 한번에 실행됩니다. 
- docent dataset

## 실행 방법
최초 실행 시 aws S3에서 docent 데이터를 다운받으므로 시간이 걸립니다.
데이터 다운로드 이후 klue, kmounlp, docent 순으로 학습을 진행합니다.

### Docker
도커 컨테이너가 필요한 경우, 아래 명령어를 통해 도커 이미지를 빌드합니다.

```bash
docker build -t {image_name} -f Dockerfile .
docker run -it --gpus all -p 8888:8888 -v {host_dir}:{container_dir} {image_name}
```

### Bash
필요 패키지를 설치한 후 `run.sh` 파일을 실행하여 한 번에 학습을 진행할 수 있습니다.

```bash
pip install -r requirements.txt
bash run.sh
```

### 개별 실행

개별적으로 학습 혹은 평가 실행이 필요한 경우 main.py 실행 시 데이터셋 이름을 지정해주면 됩니다.
단, 상기했듯이 klue와 한국해양대학교 데이터셋은 klue로 명명되었으므로, 개별 실행 시에는 klue로 호출해야 합니다.

```bash
python3 main.py --do_train --do_eval --dset_name {dataset_name}
```
- `--load_checkpoint`: 이전 학습의 checkpoint를 불러와 학습을 이어서 진행합니다.
- `--do_train`: 지정한 데이터셋으로 학습을 진행합니다.
- `--do_eval`: 지정한 데이터셋으로 평가를 진행합니다.
- `--dset_name`: 학습 및 평가를 진행할 데이터셋 이름을 지정합니다. (klue, docent)

### PLM 변경

config 파일에서 model_path_or_name 부분을 변경하면 됩니다.

default 는 `monologg/koelectra-base-v3-discriminator` 입니다.


roberta 등을 사용하는 경우, `config.yaml`의 `use_token_types`을 False로 변경해야 합니다.