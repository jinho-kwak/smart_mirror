# smart_retail_docker

## how to install
---
    sudo apt-get -y install docker.io
> docker에 sudo 권한을 주기 위하여 다음과 같은 명령어를 추가한다

    sudo usermod -aG docker $(whoami)
    sudo systemctl reboot
> 이제 sudo 를 사용하지 않고도 docker 명령을 쓸 수 있다.

    sudo apt-get install python-setuptools python-pip -y

    pip install awscli

>  aws configure 를 설정 해야하기 때문에 aws-cli를 설치한다. 
---
## docker-compose 를 설치 하기 위해 쓰는 간단한 방법
    docker run --name tmp_hello hello-world
    sudo apt-get -y install docker-compose
    docker rm tmp_hello
    docker rmi -f hello-world

> hello-world를 사용하여 compose를 인스톨 한다. 이후 사용했던 이미지와 컨테이너를 삭제한다.

---

## how to do
---
*주의사항 : 코드내의 config.py 의 EC2_inference_ip 를 수정한다*


1. AWS 키 추가하기

.env 환경변수 파일을 같은 디렉토리 내에 추가한다.   
env_example.txt 파일에 있는 것은 복사한후 해당하는 값들에 자신이 발급받은 AWS 추가한다.

2. aws configure 설정하기

aws configure에 자신이 발급받은 키로 수정한다.

    aws configure


3. 사용방법

먼저 registry에서 이미지를 pull 해야한다. 
    
    docker-compose -f docker-registry up -d
>이미지 저장소인 s3 버킷에 접근 할 수 있도록 마운트 시킨다.   
>registry 를 5000번 포트로 매핑한다.

    docker-compose up -d

>컨테이너를 실행한다. 이때 이미지가 없다면, registry에서 자동으로 이미지를 pull 한다.   

>두개의 컨테이너를 만든다. 각각의 컨테이너는 server_main.py 과 server_inference.py 를 실행한다.

>host 의 network를 그대로 사용하며 컨테이너 실행시 자동으로 프로그램이 실행된다.  


---
## docker 명령어 [컨테이너 id 또는 name] 
* run : 컨테이너 생성 및 실행
    * -i : 사용자가 입출력을 할 수 있는 상태
    * -t : 가상 터미널 환경을 활성화 한다.
    * -d : 컨테이너를 데몬프로세스로 실행한다.
    * --name : 컨테이너에 이름을 정한다.
    * -p : host와 컨테이너 사이의 포트를 연결한다.
* images : local에 pull 된 image 확인 
* ps : 실행중인 컨테이너 확인
    * -a : 모든 컨테이너 확인
* start : 컨테이너 시작
* restart : 컨테이너 재시작
* attach : 실행중인 컨테이너에 접속
* exec : 실행중인 컨테이너에서 명령을 실행한다.   
(쉘을 사용하고 싶으면 (docker exec -it [container name] /bin/bash ))
* stop : 컨테이너 정지
* rmi : 이미지 삭제
* rm : 컨테이너 삭제

---
## docker-compose 명령어
* up : docker-compose.yml 파일을 토대로 컨테이너를 실행한다.  
 (docker-compose.yml 파일 수정했을시 자동으로 새로운 docker-compose.yml 파일로 실행해준다.) 
    * -d : 컨테이너를 데몬프로세스로 실행한다.
* down : 정의되어 있는 모든 컨테이너를 정지하고 삭제한다.
* start : 실행이 멈춘 특정 컨테이너를 시작할때 사용한다.
* stop : 실행되고 있는 특정 컨테이너를 정지 시킬때 사용한다. 
* ps : docker-compose로 실행된 컨테이너들을 확인한다.
* -f : 다른 이름이나 경로의 파일을 docker-compose 설정 파일로 사용하고 싶을때
    * ex : docker-compose -f docker-registry.yml up
* logs : 컨테이너의 로그를 확일 할 때 사용한다.
    * -f : 실시간으로 추가되는 로그도 확인한다.
* config : docker-compose.yml 설정을 확일 할때 사용한다. f 옵션으로 여러개의 설정 파일을 사용할때 어떻게 적용이 되는지 확인한다.

