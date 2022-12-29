# File to build and start container for given BRANCH

# docker_helper.sh austin-dev

BRANCH="$1"
echo $BRANCH

# BRANCH="austin-dev"
# echo $BRANCH

# 0. Build an image from dockerfile
docker build -t $BRANCH-image . -f Dockerfile #&>> docker_logs.txt

# 1. STOP docker container IF EXISTS/RUNNING
docker stop $BRANCH-container

# 2. CREATE / REPLACE & 3. RUN > Start container from image
docker run --rm --name $BRANCH-container -di $BRANCH-image #&>> docker_logs.txt