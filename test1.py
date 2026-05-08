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
  --des "0425——holidayprefix全量数据_GPT2_time_tokenlen_24" \
  --is_training 1 \
  --model_id ms_ausgrid_multi_station \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token9]]24_gpt2.pt \
  --weather_read_mode torch \
  --use_prefix \
  --holiday_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --prefix_calendar_dim 18 \
  --seq_len 168 \
  --label_len 144 \
  --token_len 24 \
  --test_seq_len 168 \
  --test_label_len 144 \
  --test_pred_len 24 \
  --train_pred_len 24 \
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
  --mix_embeds \
  --use_multiscale \
  --ms_fusion weighted \
  --ms_pattern_pool 4 \
  --ms_components fine,pattern,rr''',
  '''python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token48_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 48 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16''',
  '''python run.py \
  --task_name long_term_forecast \
  --des "0425——holidayprefix全量数据_GPT2_time_tokenlen_48" \
  --is_training 1 \
  --model_id ms_ausgrid_multi_station \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token48_gpt2.pt \
  --weather_read_mode torch \
  --use_prefix \
  --holiday_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --prefix_calendar_dim 18 \
  --seq_len 336 \
  --label_len 288 \
  --token_len 48 \
  --test_seq_len 336 \
  --test_label_len 288 \
  --test_pred_len 48 \
  --train_pred_len 48 \
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
  --mix_embeds \
  --use_multiscale \
  --ms_fusion weighted \
  --ms_pattern_pool 4 \
  --ms_components fine,pattern,rr''',
  '''python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token192_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 192 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16''',
  '''python run.py \
  --task_name long_term_forecast \
  --des "0425——holidayprefix全量数据_GPT2_time_tokenlen_192" \
  --is_training 1 \
  --model_id ms_ausgrid_multi_station \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token192_gpt2.pt \
  --weather_read_mode torch \
  --use_prefix \
  --holiday_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --prefix_calendar_dim 18 \
  --seq_len 1344 \
  --label_len 1152 \
  --token_len 192 \
  --test_seq_len 1344 \
  --test_label_len 1152 \
  --test_pred_len 192 \
  --train_pred_len 192 \
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
  --mix_embeds \
  --use_multiscale \
  --ms_fusion weighted \
  --ms_pattern_pool 4 \
  --ms_components fine,pattern,rr''',
  '''python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token384_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 192 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16''',
  '''python run.py \
  --task_name long_term_forecast \
  --des "0425——holidayprefix全量数据_GPT2_time_tokenlen_384" \
  --is_training 1 \
  --model_id ms_ausgrid_multi_station \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token384_gpt2.pt \
  --weather_read_mode torch \
  --use_prefix \
  --holiday_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --prefix_calendar_dim 18 \
  --seq_len 2688 \
  --label_len 2304 \
  --token_len 384 \
  --test_seq_len 2688 \
  --test_label_len 2304 \
  --test_pred_len 384 \
  --train_pred_len 384 \
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
  --mix_embeds \
  --use_multiscale \
  --ms_fusion weighted \
  --ms_pattern_pool 4 \
  --ms_components fine,pattern,rr'''
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