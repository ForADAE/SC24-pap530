#!/bin/bash

python mem-predict/gen_sub_lstm.py > mem-predict/batch_lstm.sh
python mem-predict/gen_sub.py > mem-predict/batch.sh

bash mem-predict/batch_lstm.sh >& batch_lstm.log
bash mem-predict/batch.sh >& batch.log