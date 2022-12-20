# File to build and start container for given BRANCH

BRANCH="austin-dev"
echo $BRANCH

# Build an image from dockerfile
docker build -t $BRANCH . -f Dockerfile

# Start container from image
docker run --name $BRANCH -i $BRANCH