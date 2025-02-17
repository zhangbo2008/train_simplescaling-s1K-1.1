# 把训练代码改成cpu版本.

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        # os.environ['WANDB_PROJECT'] = self.wandb_project
        # os.environ['WANDB_ENTITY'] = self.wandb_entity
        pass

def train():
    #
    uid=0
    base_model="Qwen/Qwen2.5-0.5B-Instruct" #=======2025-02-14,17点42 这个根据自己需要, 我是离线的所以我设置自己路径.    Qwen/Qwen2.5-0.5B-Instruct
    lr=1e-5
    min_lr=0
    epochs=5
    weight_decay=1e-4 # -> the same training pipe as slurm_training
    micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
    gradient_accumulation_steps=1 # requires more GPU memory
    max_steps=-1

    tianjia=[
    f'--block_size=1000', # ========这个改小点. 太废显存. 默认32768
    f'--per_device_train_batch_size={micro_batch_size}',
    f'--per_device_eval_batch_size={micro_batch_size}',
    f'--gradient_accumulation_steps={gradient_accumulation_steps}',
    f'--num_train_epochs={epochs}',
    f'--train_file_path=simplescaling/s1K_tokenized',
    f'--model_name={base_model}',
    f'--warmup_ratio=0.05',
    f'--gradient_checkpointing=True',
    # f'--fsdp=full_shard auto_wrap',
    # f'--fsdp_config=train/fsdp_config_qwen.json',
    f'--bf16=True',
    f'--eval_strategy=no',
    f'--logging_steps=1',
    f'--save_strategy=no',
    f'--lr_scheduler_type=cosine',
    f'--learning_rate={lr}',
    f'--weight_decay={weight_decay}',
    f'--adam_beta1=0.9',
    f'--adam_beta2=0.95',
    f'--output_dir=ckpts/s1-{uid}',
    f'--save_only_model=True',]
    for i  in tianjia:
            sys.argv.append(i)
    
    
    
    
    
    
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    parser.add_argument("-f","--file",default="file")#接收这个-f参数 #添加这个可以让这个在colab中运行.
    config, args,_ = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    print('模型训练参数是',log_config)

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    print('加载完model')
    dataset = load_dataset(config.train_file_path)
    print('加载完数据.')
    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name or 'qwen' in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
