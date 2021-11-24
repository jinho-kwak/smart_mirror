#SD카드 auto_install 스크립트

#config 수정 목록 

#Interface Options -> SSH -> enable 로 변경
#Interface Options -> VNC -> enable 로 변경
#finish -> reboot

#install 목록

sudo apt upgrade
sudo apt update

#Anydesk 설치
#Anydesk install anydesk_x.x.x-x_armhf.deb 파일이 다운로드 후 실행
sudo dpkg -i anydesk_6.1.0-1_armhf.deb
sudo apt-get -y -f install

#vim 설치
sudo apt-get -y install vim

#color log 설치
pip install colorlog
#minicom 설치
sudo apt-get -y install minicom

#폰트 업데이트, 한글 깨짐현상 제거
sudo apt-get install fonts-unfonts-core


#pip 인스톨 업그레이드 , Redis 설치
sudo pip3 install --upgrade pip
sudo pip3 install redis

#opencv 설치 
pip3 install opencv-python
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libjasper-dev
sudo apt-get -y install libqtgui4
sudo apt-get -y install python3-pyqt5
sudo pip3 -y uninstall opencv-python
sudo pip3 install opencv-python==3.4.6.27

#pycryptodome 설치
pip3 install pycryptodome

#apscheduler 설치
pip3 install apscheduler

#flask 설치
sudo pip3 install flask

sudo apt upgrade
sudo apt update
