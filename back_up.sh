#!/bin/bash
now_dir_path=`pwd -P`
CUR_DATE='backup_'`date +%Y%m%d`
mkdir -p /home/ubuntu/$CUR_DATE

cp -r /home/ubuntu/smart_retail /home/ubuntu/$CUR_DATE

cd /home/ubuntu/smart_retail/
rm -rf `ls /home/ubuntu/smart_retail/ | grep -v logs | grep -v keys`
rm -rf .git

cd $now_dir_path
cp -i -rf smart_retail/* /home/ubuntu/smart_retail/
cp -i -rf smart_retail/.git /home/ubuntu/smart_retail/
