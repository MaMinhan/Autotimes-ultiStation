import subprocess
import time

GPU_ID = 0
UTIL_THRESHOLD = 10
MEM_THRESHOLD = 5000

CHECK_INTERVAL = 10
IDLE_TIME = 10  # 1分钟

commands = [
'''
python train_xgb_external_memory.py \
  --train_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/train \
  --val_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/val \
  --test_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/test \
  --out_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/output \
  --eval_max_parts 20
'''
  ]

def get_gpu_info(gpu_id):
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).decode()

    util, mem = result.strip().split("\n")[gpu_id].split(", ")
    return int(util), int(mem)


def wait_for_gpu():
    idle_seconds = 0

    while True:
        util, mem = get_gpu_info(GPU_ID)

        print(f"GPU{GPU_ID} util={util}% mem={mem}MB idle={idle_seconds}s")

        if util < UTIL_THRESHOLD and mem < MEM_THRESHOLD:
            idle_seconds += CHECK_INTERVAL
        else:
            idle_seconds = 0

        if idle_seconds >= IDLE_TIME:
            print("GPU 已空闲 5 分钟")
            return

        time.sleep(CHECK_INTERVAL)


for cmd in commands:
    wait_for_gpu()

    print("\n============================")
    print("开始执行:", cmd)
    print("============================\n")

    subprocess.call(cmd, shell=True)

print("所有任务完成")