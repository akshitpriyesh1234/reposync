#!/bin/bash
cd /home/wyk1kor/FIFTYONE_ONE_TWO/codes/rsync/child/reposync
# git status
# echo "Do you want to commit the changes? (y/n)"
# read answer
# if [ "$answer" == "y" ]; then
#     git add .
#     git commit -m "Manually syncing selected files from repoParent"
#     git push origin main
#     echo "Changes committed and pushed!"
# else
#     echo "Changes are not committed."
# fi

git status
echo "Do you want to commit the changes? (y/n)"
read answer
if [ "$answer" == "y" ]; then
    git add .
    git commit -m "Manually syncing selected files from repoParent"
    
    # Detect the current branch dynamically
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    
    git push origin "$CURRENT_BRANCH"
    echo "Changes committed and pushed to $CURRENT_BRANCH!"
else
    echo "Changes are not committed."
fi
