#!/bin/bash

DIR="$( cd "$( dirname "$0" )" && pwd -P )"
pName="server_main.py"

cd $DIR

# 로그 매일 자정마다 백업
(echo '0 0 * * * cp '$DIR'/logs/log/status.log '$DIR'/logs/log/status.log-$(date +\%Y\%m\%d); cat dev/null > '$DIR'/logs/log/status.log') | crontab -

# Start
if [ $1 = "start" ]
then
        pCount=$(ps -ef | grep $pName | grep -v grep | grep -v tail | grep -v vi |grep -v $pName.sh | grep -v $pName.log | wc -l)
        if [ $pCount -eq 0 ]
        then
                echo "start $pName..."
                nohup python3 -u $pName > /dev/null 2>&1 & echo $! > $DIR/pid/$pName.pid
        else
                echo "$pName is already running..."
        fi
# Stop
elif [ $1 = "stop" ]
then
        echo "stop $pName..."
        kill -9 `ps -ef | grep $pName | grep -v grep | grep -v tail | grep -v vi | grep -v $pName.sh | grep -v $pName.log | awk '{print $2}'`
        rm $DIR/pid/$pName.pid

else
        echo "No parameter $pName"
fi
