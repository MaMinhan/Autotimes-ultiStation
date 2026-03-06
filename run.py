import argparse                 # 命令行参数解析：让你用 python run.py --xxx 传参
import os                       # 读环境变量、拼路径（多卡/保存路径用）
import random                   # Python 随机数（用于复现）
import numpy as np              # numpy 随机数（用于复现、数据处理）
import torch                    # PyTorch 主库
import torch.distributed as dist# DDP 分布式训练（多 GPU）相关

# 不同任务对应的“实验类”(Exp)：封装 train/test 流程、模型构建、优化器等
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from exp.exp_in_context_forecasting import Exp_In_Context_Forecast


if __name__ == '__main__':  # 脚本入口（只有直接运行 run.py 才执行下面逻辑）

    # ========== 1) 固定随机种子（尽量保证可复现） ==========
    fix_seed = 2021
    random.seed(fix_seed)       # Python 随机
    torch.manual_seed(fix_seed) # PyTorch 随机（CPU）
    np.random.seed(fix_seed)    # numpy 随机

    # ========== 2) 创建 argparse 解析器 ==========
    parser = argparse.ArgumentParser(description='AutoTimes')

    # ---------------- basic config：任务/模型选择（决定训练流程走哪条分支） ----------------
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='任务类型：长预测/短预测/零样本/ICL')
    parser.add_argument('--is_training', type=int, required=True, default=1,
                        help='1=训练+测试；0=只测试（加载checkpoint）')
    parser.add_argument('--model_id', type=str, required=True, default='test',
                        help='实验ID，用于实验命名/区分不同run')
    parser.add_argument('--model', type=str, required=True, default='AutoTimes_Llama',
                        help='模型族：AutoTimes_Llama / AutoTimes_Gpt2 / AutoTimes_Opt1b')

    # ---------------- data loader：数据相关（决定用哪个Dataset、读什么文件、怎么切） ----------------
    parser.add_argument('--data', type=str, required=True, default='ETTm1',
                        help='数据集类型key：data_factory/data_loader里用它选择Dataset')
    parser.add_argument('--root_path', type=str, default='./data/ETT/',
                        help='数据根目录（csv/pt 等文件所在路径）')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='数据文件名（例如 merged.csv / ETTh1.csv）')
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv',
                        help='zero-shot时可能用不同测试文件（你的long_term一般用不到）')
    parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/outputs',
                        help='模型checkpoint保存目录')
    parser.add_argument('--drop_last',  action='store_true', default=False,
                        help='DataLoader是否丢弃最后一个不足batch的残批')
    parser.add_argument('--val_set_shuffle', action='store_false', default=True,
                        help='验证集是否shuffle（注意：store_false，写了这个flag才会变False）')
    parser.add_argument('--drop_short', action='store_true', default=False,
                        help='对太短的序列直接丢弃（TSF/ICL类更常用）')

    # 你自定义 multi-station/时间embedding 使用的 time_only.pt 路径（全局时间轴embedding）
    parser.add_argument('--time_pt_path', type=str,
                        default='/root/autodl-tmp/datasets/electricity_pt/time_only_20260129.pt')
    parser.add_argument('--weather_pt_path', type=str, default='')
    # npy 路线相关路径：仅当 args.data == "npy" 时会用到
    parser.add_argument('--npy_x_path', type=str, default='')
    parser.add_argument('--npy_y_path', type=str, default='')
    parser.add_argument('--npy_sid_path', type=str, default='')

    # multi-station里是否按站点scale（你在data_factory里读 args.ms_scale）
    parser.add_argument("--ms_scale", type=int, default=0)

    # npy 数据切分比例：仅 npy 路线用
    parser.add_argument('--npy_train_ratio', type=float, default=0.7)
    parser.add_argument('--npy_val_ratio', type=float, default=0.1)

    # npy mark embedding维度（通常是 768，对应 gpt2 hidden_size）
    parser.add_argument('--npy_mark_dim', type=int, default=768)

    # ---------------- forecasting task：窗口长度（决定样本怎么切） ----------------
    parser.add_argument('--seq_len', type=int, default=672,
                        help='输入历史长度（encoder看到的历史步数）')
    parser.add_argument('--label_len', type=int, default=576,
                        help='decoder可见的历史标签长度（teacher forcing 的历史部分）')
    parser.add_argument('--token_len', type=int, default=96,
                        help='token长度（AutoTimes用它决定time token采样粒度；同时训练时也常复用作pred_len）')

    # test 阶段可单独设置长度（但你 custom_ms 里目前只真正用到了 test_pred_len）
    parser.add_argument('--test_seq_len', type=int, default=672, help='test用的seq_len（可能未生效）')
    parser.add_argument('--test_label_len', type=int, default=576, help='test用的label_len（可能未生效）')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test用的预测长度pred_len')

    # M4 数据集的子集选择（长预测一般不用）
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='M4子集标识（Monthly/Yearly等）')

    # ---------------- model define：适配层/LLM权重路径（决定“用哪个预训练LLM”+“训练哪些层”） ----------------
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout概率（适配层/投影层等可能用到）')

    # HuggingFace LLM checkpoint路径：AutoTimes_Gpt2 时一般指向 gpt2 目录
    parser.add_argument('--llm_ckp_dir', type=str, default='/root/autodl-tmp/hf_models/gpt2',
                        help='预训练语言模型权重目录（HF格式）')

    # 适配MLP的规模：hidden_dim、层数、激活函数（控制可训练参数量/表达能力）
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='MLP hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='MLP layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='MLP activation')

    # ---------------- optimization：训练超参（决定优化过程、lr调度、早停等） ----------------
    parser.add_argument('--num_workers', type=int, default=10,
                        help='DataLoader worker数量（越大越快但更占CPU/内存）')
    parser.add_argument('--itr', type=int, default=1,
                        help='重复实验次数（会循环跑itr次并编号ii）')
    parser.add_argument('--train_epochs', type=int, default=10, help='最大训练epoch数')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3,
                        help='EarlyStopping耐心值：验证集多少轮不提升就停')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='初始学习率（核心超参之一）')
    parser.add_argument('--des', type=str, default='test',
                        help='实验描述字符串：写进setting名里')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数类型（一般MSE）')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='学习率调整策略名（Exp里会用它决定怎么衰减）')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='是否开启混合精度AMP（省显存/更快）')

    # cosine 调度器开关：只有写了 --cosine 才启用
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='是否使用CosineAnnealingLR')
    parser.add_argument('--tmax', type=int, default=10,
                        help='cosine调度的T_max参数（一个周期长度）')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='权重衰减（L2正则）')

    # mix_embeds：AutoTimes里通常表示“把不同embedding混合/拼接”（时间/外生等）
    parser.add_argument('--mix_embeds', action='store_true', default=False,
                        help='是否启用embedding混合（你后续加外生变量可能会用）')

    # 只测试模式：指定测试加载的checkpoint路径
    parser.add_argument('--test_dir', type=str, default='./test', help='只测试时checkpoint目录')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='只测试时checkpoint文件名')

    # ---------------- GPU：单卡/多卡开关 ----------------
    parser.add_argument('--gpu', type=int, default=0, help='单卡训练用哪张GPU')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='是否启用多GPU（DDP）')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='可视化开关（run.py里没用，可能Exp里用）')
    parser.add_argument('--resume', action='store_true', default=False,
                    help='是否从已有checkpoint加载后继续训练')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='要继续训练的checkpoint完整路径')
    # 解析命令行参数：把 --xxx 转成 args.xxx
    args = parser.parse_args()
    
    # ========== 3) 多卡DDP初始化（只有 --use_multi_gpu 时才执行） ==========
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")   # 主节点IP（torchrun会设）
        port = os.environ.get("MASTER_PORT", "64209")     # 通信端口
        hosts = int(os.environ.get("WORLD_SIZE", "8"))    # 总进程数（通常=总GPU数）
        rank = int(os.environ.get("RANK", "0"))           # 全局rank（0..WORLD_SIZE-1）
        local_rank = int(os.environ.get("LOCAL_RANK", "0")) # 本机GPU编号
        gpus = torch.cuda.device_count()                  # 当前机器可见GPU数量
        args.local_rank = local_rank                      # 保存local_rank给Exp/日志用

        print(ip, port, hosts, rank, local_rank, gpus)    # 打印DDP信息便于排查

        # 初始化进程组：backend=nccl（GPU通信最快）
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=hosts,
            rank=rank
        )

        # 每个进程绑定到自己的GPU
        torch.cuda.set_device(local_rank)

    # ========== 4) 根据 task_name 选择实验类（不同任务不同train/test逻辑） ==========
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'zero_shot_forecast':
        Exp = Exp_Zero_Shot_Forecast
    elif args.task_name == 'in_context_forecast':
        Exp = Exp_In_Context_Forecast
    else:
        Exp = Exp_Long_Term_Forecast  # 默认兜底
    # ========== 5) 训练模式：训练+测试 ==========
    if args.is_training:
        for ii in range(args.itr):  # 重复实验itr次（便于跑多个seed/配置）
            exp = Exp(args)         # 构造实验对象：里面会 build model / build dataloader 等

            # setting：实验唯一标识字符串（用于 checkpoint 子目录命名 + 日志标识）
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,         # 任务名
                args.model_id,          # 你给的实验ID
                args.model,             # 模型族
                args.data,              # 数据key（决定Dataset）
                args.seq_len,           # sl
                args.label_len,         # ll
                args.token_len,         # tl（token粒度标记；你做消融时非常重要）
                args.learning_rate,     # lr
                args.batch_size,        # batch
                args.weight_decay,      # wd
                args.mlp_hidden_dim,    # 适配MLP隐藏维度
                args.mlp_hidden_layers, # 适配MLP层数
                args.cosine,            # 是否cosine调度
                args.mix_embeds,        # 是否mix embedding
                args.des,               # 你填的描述
                args.ms_scale,           # multi-station是否按站点scale
                args.train_epochs,              # epoch数（有时也写在setting里）
                ii                      # 第几次重复实验
            )

            # 只有主进程打印，避免多卡刷屏；单卡直接打印
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            exp.train(setting)  # 训练：内部会跑epoch、验证、early stopping、保存checkpoint

            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

            exp.test(setting)   # 测试：读取训练好的checkpoint，在test集上评估指标并输出结果

            torch.cuda.empty_cache()  # 清理显存碎片（不一定必要，但有时能缓解显存抖动）

    # ========== 6) 只测试模式：不训练，直接加载指定checkpoint测试 ==========
    else:
        ii = 0  # 只测时固定编号0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.des, ii
        )

        exp = Exp(args)             # 构造实验对象（会build model）
        exp.test(setting, test=1)   # test=1 通常表示“从 args.test_dir/args.test_file_name 加载”
        torch.cuda.empty_cache()
