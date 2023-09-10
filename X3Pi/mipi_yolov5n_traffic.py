#!/usr/bin/env python3
import os
import numpy as np
from time import time, sleep
import multiprocessing
from threading import BoundedSemaphore

import Hobot.GPIO as GPIO
# Camera API libs
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn

from postprocess import postprocess

image_counter = multiprocessing.Value("i", 0)
last_traffic = multiprocessing.Value("i", -1)
last_time = multiprocessing.Value("d", 0)

left_pin = 11
front_pin = 13
right_pin = 15


def sensor_reset_shell():
    os.system("echo 19 > /sys/class/gpio/export")
    os.system("echo out > /sys/class/gpio/gpio19/direction")
    os.system("echo 0 > /sys/class/gpio/gpio19/value")
    sleep(0.2)
    os.system("echo 1 > /sys/class/gpio/gpio19/value")
    os.system("echo 19 > /sys/class/gpio/unexport")
    os.system("echo 1 > /sys/class/vps/mipi_host0/param/stop_check_instart")


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


class ParallelExector(object):
    def __init__(self, counter, _traffic, _time, parallel_num=4):
        global image_counter
        global last_traffic
        global last_time
        image_counter = counter
        last_time = _time
        last_traffic = _traffic
        self.parallel_num = parallel_num
        if parallel_num != 1:
            self._pool = multiprocessing.Pool(processes=self.parallel_num,
                                              maxtasksperchild=5)
            self.workers = BoundedSemaphore(self.parallel_num)

    def infer(self, output):
        if self.parallel_num == 1:
            run(output)
        else:
            self.workers.acquire()
            self._pool.apply_async(func=run,
                                   args=(output, ),
                                   callback=self.task_done,
                                   error_callback=print)

    def task_done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        self.workers.release()

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()


def run(outputs):
    # 多进程的进程句柄
    global image_counter
    global last_time
    global last_traffic
    # Do post process
    detection = postprocess(outputs, model_hw_shape=(672, 672))
    traffic_dict = {
        '': -1,
        'down_stair': 0,
        'up_stair': 0,
        'green_light': 1,
        'red_light': 2,
        'yellow_light': 3
    }
    traffic = traffic_dict[detection]
    # print(f'traffic: {traffic}')

    # fps timer and counter
    with image_counter.get_lock():
        image_counter.value += 1
    if image_counter.value == 100:
        finish_time = time()
        print(
            f"Total time cost for 100 frames: {finish_time - start_time}, fps: {100/(finish_time - start_time)}"
        )

    # 融合光电传感器信息
    if traffic == -1:
        if GPIO.input(left_pin) == GPIO.HIGH:
            traffic = 4
        elif GPIO.input(front_pin) == GPIO.HIGH:
            traffic = 5
        elif GPIO.input(right_pin) == GPIO.HIGH:
            traffic = 6

    # 播放对应语音
    # print(f'last_traffic: {last_traffic.value}, time: {time()}, last_time: {last_time.value}')
    if traffic != -1:
        if last_traffic.value == -1:
            with last_traffic.get_lock():
                last_traffic.value = traffic
            with last_time.get_lock():
                last_time.value = time()
            os.system(f'mpg123 -q -f 65536 {traffic}.mp3')
        elif time() - last_time.value > 3 and traffic != last_traffic.value:  # 间隔3s以上且信息出现变化才提醒
            with last_traffic.get_lock():
                last_traffic.value = traffic
            with last_time.get_lock():
                last_time.value = time()
            os.system(f'mpg123 -q -f 65536 {traffic}.mp3')


if __name__ == '__main__':
    models = dnn.load('./yolov5n_672x672_nv12.bin')
    print("--- model input properties ---")
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    print("--- model output properties ---")
    # 打印输出 tensor 的属性
    for output in models[0].outputs:
        print_properties(output.properties)

    # Camera API, get camera object
    cam = srcampy.Camera()

    # get model info
    h, w = get_hw(models[0].inputs[0].properties)
    input_shape = (h, w)
    sensor_reset_shell()
    # Open f37 camera
    # For the meaning of parameters, please refer to the relevant documents of camera
    cam.open_cam(0, 1, 30, [w], [h])

    # fps timer and counter
    start_time = time()
    # image_counter = multiprocessing.Value("i", 0)
    # last_traffic = multiprocessing.Value("i", -1)
    # last_time = multiprocessing.Value("i", 0)

    # post process parallel executor
    parallel_exe = ParallelExector(image_counter, last_traffic, last_time)

    GPIO.setmode(GPIO.BOARD)  # 设置管脚编码模式为硬件编号 BOARD
    GPIO.setup([left_pin, front_pin, right_pin], GPIO.IN)  # 设置为输入模式

    try:
        while True:
            # image_counter += 1
            # Get image data with shape of 672x672 nv12 data from camera
            img = cam.get_img(2, 672, 672)

            # # Convert to numpy
            img = np.frombuffer(img, dtype=np.uint8)

            # Forward
            outputs = models[0].forward(img)

            output_array = []
            for item in outputs:
                output_array.append(item.buffer)
            parallel_exe.infer(output_array)
    finally:
        cam.close_cam()
        GPIO.cleanup()
