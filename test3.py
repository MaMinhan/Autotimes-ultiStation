import subprocess
import time

GPU_ID = 0
UTIL_THRESHOLD = 10
MEM_THRESHOLD = 1000
CHECK_INTERVAL = 10
IDLE_TIME = 60  # GPU连续空闲60秒后开始下一个任务

# =========================
# 第一批：只扫 learning_rate
# =========================
stage1_learning_rates = ["5e-5", "2e-4"]

# =========================
# 第二批：固定最优 learning_rate，再扫指定的 MLP 组合
# 这里只跑两组：
# (128, 1) 和 (512, 3)
# =========================
best_lr = "1e-4"
stage2_mlp_settings = [
    (128, 1),
    (512, 3),
]

# =========================
# 运行哪一批
# 可选：
#   RUN_STAGE = 1   -> 只跑第一批
#   RUN_STAGE = 2   -> 只跑第二批
#   RUN_STAGE = 12  -> 两批都跑
# =========================
RUN_STAGE = 2


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
            print(f"GPU 已空闲 {IDLE_TIME} 秒，开始执行下一个任务。")
            return

        time.sleep(CHECK_INTERVAL)


def build_base_command(des, learning_rate, mlp_hidden_dim, mlp_hidden_layers):
    cmd = f'''python run.py \
  --task_name long_term_forecast \
  --des "{des}" \
  --is_training 1 \
  --model_id ms_ausgrid_multi_station \
  --model AutoTimes_Gpt2 \
  --data custom_ms \
  --root_path /root/autodl-tmp/datasets/SelfMadeAusgridData \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --use_time 0 \
  --use_weather 1 \
  --weather_pt_path /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/0406_time+temp_token96_gpt2.pt \
  --weather_read_mode torch \
  --use_prefix \
  --holiday_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --prefix_calendar_dim 18 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 192 \
  --train_pred_len 192 \
  --batch_size 576 \
  --learning_rate {learning_rate} \
  --num_workers 10 \
  --gpu 0 \
  --checkpoints /root/autodl-tmp/checkpoints \
  --ms_scale 1 \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --mlp_hidden_dim {mlp_hidden_dim} \
  --mlp_hidden_layers {mlp_hidden_layers} \
  --train_epochs 3 \
  --patience 1 \
  --mark_input_dim 768 \
  --use_amp \
  --mix_embeds'''
    return cmd


def build_stage1_commands():
    commands = []
    for lr in stage1_learning_rates:
        des = f"0407_holidayprefix_full_GPT2_time_temp_stage1_lr{lr}"
        cmd = build_base_command(
            des=des,
            learning_rate=lr,
            mlp_hidden_dim=256,
            mlp_hidden_layers=2
        )
        commands.append(cmd)
    return commands


def build_stage2_commands():
    commands = []
    for mlp_dim, mlp_layers in stage2_mlp_settings:
        des = f"0407_holidayprefix_full_GPT2_time_temp_stage2_lr{best_lr}_mlpdim{mlp_dim}_mlplayers{mlp_layers}"
        cmd = build_base_command(
            des=des,
            learning_rate=best_lr,
            mlp_hidden_dim=mlp_dim,
            mlp_hidden_layers=mlp_layers
        )
        commands.append(cmd)
    return commands


def main():
    commands = []

    if RUN_STAGE == 1:
        commands = build_stage1_commands()
    elif RUN_STAGE == 2:
        commands = build_stage2_commands()
    elif RUN_STAGE == 12:
        commands = build_stage1_commands() + build_stage2_commands()
    else:
        raise ValueError("RUN_STAGE 只能是 1 / 2 / 12")

    print(f"总任务数: {len(commands)}")

    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] 等待 GPU 空闲...")
        wait_for_gpu()

        print("\n============================")
        print(f"开始执行第 {i}/{len(commands)} 个任务:")
        print(cmd)
        print("============================\n")

        ret = subprocess.call(cmd, shell=True)

        if ret != 0:
            print(f"任务 {i} 执行失败，返回码: {ret}")
        else:
            print(f"任务 {i} 执行完成。")

    print("所有任务完成")


if __name__ == "__main__":
    main()