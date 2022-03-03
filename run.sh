pkill -9 python3
DIR="$( cd "$( dirname "$0" )" && pwd -P )"
cd $DIR
echo $DIR

# 로그 매일 자정마다 백업
(echo '0 0 * * * cp '$DIR'/logs/log/status.log '$DIR'/logs/log/status.log-$(date +\%Y\%m\%d); cat dev/null > '$DIR'/logs/log/status.log') | crontab -

nohup python3 -u server_main.py > /dev/null 2>&1 &
nohup python3 -u server_inference.py > /dev/null 2>&1 &
