import serial
import time
import numpy as np
import requests

SERVER = '172.24.68.171'

regdict = {
    'ID' : 1000,
    'baudrate' : 1001,
    'clearErr' : 1004,
    'forceClb' : 1009,
    'angleSet' : 1486,
    'forceSet' : 1498,
    'speedSet' : 1522,
    'angleAct' : 1546,
    'forceAct' : 1582,
    'errCode' : 1606,
    'statusCode' : 1612,
    'temp' : 1618,
    'actionSeq' : 2320,
    'actionRun' : 2322
}


def openSerial(port, baudrate):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    return ser


def writeRegister(ser, id, add, num, val):
    bytes = [0xEB, 0x90]
    bytes.append(id) # id
    bytes.append(num + 3) # len
    bytes.append(0x12) # cmd
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF) # add
    for i in range(num):
        bytes.append(val[i])
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    ser.flush() 
    # ser.read_all() # 把返回帧读掉，不处理


def readRegister(ser, id, add, num, mute=False):
    bytes = [0xEB, 0x90]
    bytes.append(id) # id
    bytes.append(0x04) # len
    bytes.append(0x11) # cmd
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF) # add
    bytes.append(num)
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    recv = ser.read_all()
    if len(recv) == 0:
        return []
    num = (recv[3] & 0xFF) - 3
    val = []
    for i in range(num):
        val.append(recv[7 + i])
    if not mute:
        print('读到的寄存器值依次为：', end='')
        for i in range(num):
            print(val[i], end=' ')
        print()
    return val


def write6(ser, id, str, val):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet':
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        writeRegister(ser, id, regdict[str], 12, val_reg)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')


def read6(ser, id, str):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'angleAct' or str == 'forceAct':
        val = readRegister(ser, id, regdict[str], 12, True)
        if len(val) < 12:
            print('没有读到数据')
            return
        val_act = []
        for i in range(6):
            val_act.append((val[2*i] & 0xFF) + (val[1 + 2*i] << 8))
        print('读到的值依次为：', end='')
        for i in range(6):
            print(val_act[i], end=' ')
        print()
    elif str == 'errCode' or str == 'statusCode' or str == 'temp':
        val_act = readRegister(ser, id, regdict[str], 6, True)
        if len(val_act) < 6:
            print('没有读到数据')
            return
        print('读到的值依次为：', end='')
        for i in range(6):
            print(val_act[i], end=' ')
        print()
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'temp\'')


if __name__ == '__main__':
    print('open port')
    ser_r = openSerial('/dev/ttyUSB1', 115200)
    ser_l = openSerial('/dev/ttyUSB0', 115200)
    write6(ser_r, 1, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
    write6(ser_r, 1, 'forceSet', [500, 500, 500, 500, 500, 500])
    write6(ser_r, 1, 'angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
    write6(ser_l, 1, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
    write6(ser_l, 1, 'forceSet', [500, 500, 500, 500, 500, 500])
    write6(ser_l, 1, 'angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
    time.sleep(0.5)
    finger_cmd_r = np.array([-1]*6)
    finger_cmd_l = np.array([-1]*6)
    finger_angle_l_buf = []
    finger_angle_r_buf = []
    buffer_len = 4

    while True:
        start_time = time.time()
        print("get url r")
        pose_res_r = requests.get(f'http://{SERVER}:8080/get_right_hand_pose')
        print("get url l")
        pose_res_l = requests.get(f'http://{SERVER}:8080/get_left_hand_pose')
        print("get json")
        requested_motion_r = pose_res_r.json()
        requested_motion_l = pose_res_l.json()
        print(requested_motion_l)
        print(requested_motion_r)

        # finger_angle_r = np.array(requested_motion_r['hand_pose'])
        finger_angle_r_buf.append(requested_motion_r['hand_pose'])
        if len(finger_angle_r_buf) > buffer_len:
            finger_angle_r_buf.pop(0)
        finger_angle_r = np.mean(finger_angle_r_buf,axis=0)

        finger_cmd_r[:4] = (np.clip((finger_angle_r[:4]-30)/(160-30)*1000,0,1000)).astype(int)  # range 30-160 30-close-cmd:0
        finger_cmd_r[4] = (np.clip((finger_angle_r[4]-(110))/(170-(110))*1000,0,1000)).astype(int)  # thumb close: 90 open: 140
        finger_cmd_r[-1] = (np.clip((finger_angle_r[-1]-65)/(85-65)*1000,0,1000)).astype(int)  # thumb lateral close:60 open:80 range 10-30 30-close-cmd:0 
        
        # finger_angle_l = np.array(requested_motion_l['hand_pose'])
        finger_angle_l_buf.append(requested_motion_l['hand_pose'])
        if len(finger_angle_l_buf) > buffer_len:
            finger_angle_l_buf.pop(0)
        finger_angle_l = np.mean(finger_angle_l_buf,axis=0)

        finger_cmd_l[:4] = (np.clip((finger_angle_l[:4]-30)/(160-30)*1000,0,1000)).astype(int)  # range 30-160 30-close-cmd:0
        finger_cmd_l[4] = (np.clip((finger_angle_l[4]-(110))/(170-(110))*1000,0,1000)).astype(int)  # thumb close: 90 open: 140
        finger_cmd_l[-1] = 1000-(np.clip((finger_angle_l[-1]-85)/(110-85)*1000,0,1000)).astype(int)  # thumb lateral close:60 open:80 range 10-30 30-close-cmd:0 

        # finger_cmd[:4] = np.array(requested_motion['finger_angle'][:4])
        write6(ser_r, 1, 'angleSet', finger_cmd_r)
        write6(ser_l, 1, 'angleSet', finger_cmd_l)
        print('loop time:', time.time() - start_time)  #loop time: 0.0315 if read6 loop time: 0.0156 if write only
        time.sleep(max(0, 0.033 - (time.time() - start_time)))

    # read6(ser, 1, 'angleAct')
    # read6(ser, 1, 'errCode')