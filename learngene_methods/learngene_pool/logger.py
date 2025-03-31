import logging
import os
from pathlib import Path
import json
from datetime import datetime
from params import args


dt = datetime.now()
dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
_LOG_FMT = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'

# distill保存路径 ./train_results/distill/args.teacher_model-args.student_model/'LearngenePool_args.loss_pos
# scratch保存路径 ./train_results/scratch/small/args.stitch_id
output_dir = Path(args.output_dir)
# output_dir = output_dir.joinpath('tiny', str(args.stitch_id))
# output_dir = output_dir.joinpath(args.teacher_model+'-'+args.student_model, 'LearngenePool_'+args.loss_pos)
# output_dir = output_dir.joinpath('finetune')
output_dir.mkdir(parents=True, exist_ok=True)
setattr(args, 'output_dir', str(output_dir))


# 设置Log格式
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
logger = logging.getLogger('Training_Log')
logger.setLevel(level=logging.DEBUG)  # 记录DEBUG以上的日志信息，包括（debug, info, warning, error, critical）
formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)


file_handler = logging.FileHandler(output_dir.joinpath('Training_Log.log'))  # 用于保存在log文件中
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler() # 用于输出在终端
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
# 如果存在checkpoint.pth，则将其路径给args.resume
checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
if os.path.exists(checkpoint_path) and not args.resume:
    setattr(args, 'resume', checkpoint_path)

with open(os.path.join(args.output_dir, 'args.json'), 'w+') as f:
    json.dump(args, f, indent=4)  # indent是json格式缩进4个space，便于肉眼查看
    
