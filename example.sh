#!/bin/bash

# Train - four tasks
python3 run.py --approach ehat --fourTasks --lazy_loading

# Predict - four tasks
python3 predict.py --approach singleTask --path \
./experiments/exp020/weights/best_task0_e2,\
./experiments/exp020/weights/best_task1_e9,\
./experiments/exp020/weights/best_task2_e5,\
./experiments/exp020/weights/best_task3_e12

