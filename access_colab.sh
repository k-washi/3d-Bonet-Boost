#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open "https://colab.research.google.com/drive/1m0cyB0_JpfLUr4955llrboSNWaCyouGX?hl=ja#scrollTo=Y3zQ_I5Qg5tB"
  sleep 1800
done