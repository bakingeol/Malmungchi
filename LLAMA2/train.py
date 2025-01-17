from peft import LoraConfig, get_peft_model
from accelerate.utils import DistributedType
from datasets import Dataset
import transformers
from datasets import load_dataset
import json
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import os

import argparse

tr_path = 'nikluge-sc-2023-train.jsonl'
dev_path = 'nikluge-sc-2023-dev.jsonl'
test_path = 'nikluge-sc-2023-test.jsonl'
dataset = load_dataset("json", data_files={"train": tr_path,
                                           #    "validation": dev_path
                                           })
dataset2 = load_dataset("json", data_files={"dev": dev_path,
                                            #    "validation": dev_path
                                            })


device = 'cuda' if torch.cuda.is_available else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0)
parser.add_argument('--model_id', default='beomi/llama-2-ko-7b')
parser.add_argument('--padding_side', default='left')
parser.add_argument('--batch_size', default=40)
parser.add_argument('--lr', default=1e-5)
parser.add_argument('--seed', default=42)
parser.add_argument('--r', default=8)
parser.add_argument('--alpha', default=32)
parser.add_argument('--lora_dropout', default=0.05)
parser.add_argument('--bias', default='none')
parser.add_argument('--save_steps', default=20)  # 300
parser.add_argument('--output_dir', default='./output_dir')
parser.add_argument('--gradient_accumulation_steps', default=1)
parser.add_argument('--epochs', default=1)
parser.add_argument('--logging_steps', default=1)
parser.add_argument('--train_mode', choices=['lora', 'full'])
parser.add_argument('--ckpt_full_path', default='')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_id, padding_side=args.padding_side)
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
special_tokens = {
    'pad_token': DEFAULT_PAD_TOKEN,
    'eos_token': DEFAULT_EOS_TOKEN,
    'bos_token': DEFAULT_BOS_TOKEN,
    'unk_token': DEFAULT_UNK_TOKEN,
}
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
)
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
for name, param in model.named_parameters():
    print(name, param.data)
    print('-'*50)
    if '1' in name:
        break
if args.train_mode == 'lora':
    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.ckpt_full_path)
    model.load_state_dict(state_dict)

    def find_all_linear_names(model):
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split(".")
                lora_module_names.add(
                    names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")

        return list(lora_module_names)

    target_module = find_all_linear_names(model)
    config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_module,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()


for name, param in model.named_parameters():
    print(name, param.data)
    print('*'*50)
    if '1' in name:
        break


def generate_prompt(data_point):
    return f'''
        32토큰 이하로 두 문장 사이에 올 자연스러운 문장을 작성하세요 -\n\n
        순서: 문장1, 문장2, 문장3 \n\n
        문장1 : {data_point["input"]["sentence1"]} \n\n
        문장3 : {data_point["input"]["sentence3"]} \n\n
        문장2 : {data_point["output"]}
        '''.strip()


def gen_token_prompt(data_point):
    prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(
        prompt, padding='max_length', max_length=200, truncation=True, return_tensors='pt')
    tokenized_full_prompt['input_ids'] = tokenized_full_prompt['input_ids'].squeeze(
    )
    tokenized_full_prompt['attention_mask'] = tokenized_full_prompt['attention_mask'].squeeze()
    return tokenized_full_prompt


ds = {
    'id': dataset2['dev']['id'][14000:],  # dataset['train']['id'] +
    'input': dataset2['dev']['input'][14000:],  # dataset['train']['input'] +
    'output': dataset2['dev']['output'][14000:]  # dataset['train']['output'] +
}

data = Dataset.from_dict(ds)

dataset = data.shuffle(seed=args.seed).map(
    gen_token_prompt, remove_columns=data.column_names)


training_args = transformers.TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    logging_steps=args.logging_steps,
    output_dir=args.output_dir,
    deepspeed='ds_config.json',
    report_to='tensorboard',
    save_strategy='steps',
    save_steps=args.save_steps,
    seed=args.seed
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()
