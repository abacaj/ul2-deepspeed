import torch
import os, psutil
import deepspeed
import torch.distributed as dist
from threading import Timer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, set_seed
from transformers.models.t5.modeling_t5 import T5Block

def load_model(conf):
    model_name = "google/flan-ul2"
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    infer_dtype = torch.float16

    #with deepspeed.OnDevice(dtype=infer_dtype, device="meta"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=infer_dtype)
    model = model.eval()
    model = deepspeed.init_inference(model, config=conf)

    return model.module, tokenizer


def get_gpu_mem(device_id):
    return round(torch.cuda.memory_allocated(device_id) / 1024**2, 2)


def log(rank, *msg):
    if rank != 0:
        return
    print(*msg)


def monitor_memory(rank, interval=30):
    def log_mem():
        monitor_memory(rank, interval)
        process = psutil.Process(os.getpid())
        cpu_mem = round(process.memory_info().rss / 1024**2, 2)
        gpu_mem = get_gpu_mem(rank)
        print(f"rank: {rank}, cpu: {cpu_mem}MB, gpu: {gpu_mem}MB")

    Timer(interval, log_mem).start()


if __name__ == "__main__":
    set_seed(42)
    deepspeed.init_distributed("nccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    conf = {
        "max_tokens": 1024,
        "injection_policy": {T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')},
        "dtype": torch.float16,
        "enable_cuda_graph": False,
        "tensor_parallel": {"tp_size": world_size},
    }

    log(local_rank, "Loading model...")
    monitor_memory(local_rank)

    model, tokenizer = load_model(conf)
    log(local_rank, "Model loaded")