import yaml
import argparse
import os

def update_yaml_with_args(yaml_file, args):
    # 检查yaml文件是否存在
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"The YAML file '{yaml_file}' was not found.")
    
    # 读取现有的yaml文件
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # 更新yaml文件的配置
    config['base_model'] = args.base_model
    config['data_path'] = args.data_path
    config['output_dir'] = args.output_dir
    config['batch_size'] = args.batch_size
    config['micro_batch_size'] = args.micro_batch_size
    config['num_epochs'] = args.num_epochs
    config['learning_rate'] = args.learning_rate
    config['cutoff_len'] = args.cutoff_len
    config['val_set_size'] = args.val_set_size
    config['lora_r'] = args.lora_r
    config['lora_alpha'] = args.lora_alpha
    config['lora_dropout'] = args.lora_dropout
    config['lora_target_modules'] = args.lora_target_modules
    config['train_on_inputs'] = args.train_on_inputs
    config['group_by_length'] = args.group_by_length

    # 将更新后的配置写入yaml文件
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(config, file)

def parse_args():
    parser = argparse.ArgumentParser(description="Update YAML config based on command line arguments.")
    
    # 添加 --method 参数来指定方法名，实际路径通过这个参数动态决定
    parser.add_argument('--method', type=str, required=True, help="Name of the method (used to find the corresponding YAML file)")
    
    # 其他参数与之前相同
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--micro_batch_size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--cutoff_len', type=int, required=True)
    parser.add_argument('--val_set_size', type=int, required=True)
    parser.add_argument('--lora_r', type=int, required=True)
    parser.add_argument('--lora_alpha', type=int, required=True)
    parser.add_argument('--lora_dropout', type=float, required=True)
    parser.add_argument('--lora_target_modules', type=str, required=True)
    parser.add_argument('--train_on_inputs', action='store_true')
    parser.add_argument('--group_by_length', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 根据 --method 参数来动态决定配置文件路径
    method_folder = args.method
    yaml_file = f'./learngene-methods/{method_folder}/config.yaml'

    # 将命令行参数更新到yaml文件
    try:
        update_yaml_with_args(yaml_file, args)
        print(f"YAML file '{yaml_file}' has been updated.")
    except Exception as e:
        print(f"Error: {e}")
