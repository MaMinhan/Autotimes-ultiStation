import subprocess
import time
import psutil

GPU_ID = 0

# ===== GPU 空闲阈值 =====
UTIL_THRESHOLD = 10      # GPU利用率低于10%
MEM_THRESHOLD = 5000     # GPU显存占用低于5000MB

# ===== CPU 空闲阈值 =====
CPU_THRESHOLD = 40       # CPU利用率低于30% 认为空闲

# ===== 检查间隔与连续空闲时间 =====
CHECK_INTERVAL = 10
IDLE_TIME = 20           # 连续60秒都空闲才启动下一个任务

commands = [
    '''python run.py \
  --task_name long_term_forecast \
  --des "0413多尺度+[time+temp(only_average)+humidity.pt]+holidayprefix预测" \
  --is_training 1 \
  --model_id ms_aus10 \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_humidity_token96_gpt2_0413.pt \
  --use_prefix \
  --holiday_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --prefix_calendar_dim 18 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --train_pred_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 600 \
  --learning_rate 1e-4 \
  --num_workers 10 \
  --gpu 0 \
  --checkpoints /root/autodl-tmp/checkpoints \
  --ms_scale 1 \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --mlp_hidden_dim 256 \
  --mlp_hidden_layers 2 \
  --train_epochs 3 \
  --patience 1 \
  --mark_input_dim 768 \
  --use_amp \
  --mix_embeds \
  --use_amp \
  --use_multiscale \
  --ms_fusion weighted \
  --ms_pattern_pool 4 \
  --ms_components fine,pattern,rr\
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

    lines = result.strip().split("\n")
    util, mem = lines[gpu_id].split(", ")
    return int(util), int(mem)


def get_cpu_info():
    """
    获取整体CPU利用率
    interval=1 表示采样1秒，更稳定一些
    """
    cpu_util = psutil.cpu_percent(interval=1)
    return cpu_util


def wait_for_resources():
    idle_seconds = 0

    while True:
        gpu_util, gpu_mem = get_gpu_info(GPU_ID)
        cpu_util = get_cpu_info()

        gpu_idle = (gpu_util < UTIL_THRESHOLD and gpu_mem < MEM_THRESHOLD)
        cpu_idle = (cpu_util < CPU_THRESHOLD)

        print(
            f"GPU{GPU_ID} util={gpu_util}% mem={gpu_mem}MB | "
            f"CPU util={cpu_util}% | idle={idle_seconds}s"
        )

        if gpu_idle and cpu_idle:
            idle_seconds += CHECK_INTERVAL
        else:
            idle_seconds = 0

        if idle_seconds >= IDLE_TIME:
            print("CPU 和 GPU 都已连续空闲，开始下一个任务")
            return

        time.sleep(CHECK_INTERVAL)


for i, cmd in enumerate(commands, 1):
    print(f"\n等待资源空闲后启动任务 {i}/{len(commands)} ...")
    wait_for_resources()

    print("\n============================")
    print(f"开始执行任务 {i}/{len(commands)}:")
    print(cmd)
    print("============================\n")

    ret = subprocess.call(cmd, shell=True)

    if ret != 0:
        print(f"任务 {i} 执行失败，返回码: {ret}")
    else:
        print(f"任务 {i} 执行完成")

print("所有任务完成")