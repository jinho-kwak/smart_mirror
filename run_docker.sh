echo killing old docker processes!
cd /home/ubuntu/smart_mirror
docker-compose rm -fs

echo building docker containers
docker-compose up --build -d
