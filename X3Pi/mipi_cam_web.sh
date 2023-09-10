#! /usr/bin/env bash

cd /app/ai_inference/05_web_display_camera_sample/
# 如果tcp80端口被占用, 用`lsof -i:80`找到进程, `kill -9 <pid>`
sudo sh ./start_nginx.sh
sudo python3 ./mipi_camera_web.py 