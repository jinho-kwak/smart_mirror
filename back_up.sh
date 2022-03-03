#!/bin/bash
now_dir_path=`pwd -P`
CUR_DATE='backup_'`date +%Y%m%d`
mkdir -p /home/ubuntu/$CUR_DATE

cp -r /home/ubuntu/smart_mirror /home/ubuntu/$CUR_DATE

cd /home/ubuntu/smart_mirror/
rm -rf `ls /home/ubuntu/smart_mirror/ | grep -v logs | grep -v keys`
rm -rf .git

cd $now_dir_path
cp -i -rf smart_mirror/* /home/ubuntu/smart_mirror/
cp -i -rf smart_mirror/.git /home/ubuntu/smart_mirror/
