BRANCH=${1:-agenica-dev} # take 0th argument if given, else use default
echo $BRANCH

# Build container
devcontainer build --image-name $BRANCH --workspace-folder agencia

# Open VSCode with attached container
devcontainer open agencia