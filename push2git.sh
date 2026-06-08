#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./push2git.sh \"your commit message\""
  exit 1
fi

git add -A
git commit -m "$1"
git push origin main