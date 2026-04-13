import subprocess
import time

GPU_ID = 0
UTIL_THRESHOLD = 10
MEM_THRESHOLD = 5000

CHECK_INTERVAL = 10
IDLE_TIME = 10  # 1分钟

commands = [
'''
python run.py \
  --task_name long_term_forecast \
  --des "0407——多尺度全量数据_GPT2_time_temp" \
  --is_training 1 \
  --model_id ms_ausgrid_multi_station \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/language_temperature_0413_gpt2.pt \
  --weather_read_mode torch \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --train_pred_len 96 \
  --batch_size 576 \
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
  --mix_embeds 
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