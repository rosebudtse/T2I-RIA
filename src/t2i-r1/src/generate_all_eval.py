"""
Usage:
    # 单卡
    python generate_all_eval.py \
        --model_path outputs/train_g8_1k6_full/checkpoint-800 \
        --save_root /mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results/finetuned \
        --num_generation 10 \
        --skip_existing

    # 多卡数据并行（推荐用 run_generate.sh 包装）
    torchrun --nproc_per_node=2 generate_all_eval.py \
        --model_path outputs/train_g8_1k6_full/checkpoint-800 \
        --save_root .../eval_results/finetuned \
        --num_generation 10 --skip_existing
"""

import torch
import torch.distributed as dist
import numpy as np
import os
import argparse
import random
import time
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from tqdm import tqdm


CATEGORIES = [
    ("color",       "color_val.txt"),
    ("shape",       "shape_val.txt"),
    ("texture",     "texture_val.txt"),
    ("spatial",     "spatial_val.txt"),
    ("non_spatial", "non_spatial_val.txt"),
    ("complex",     "complex_val.txt"),
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def generate_images_for_prompt(
    mmgpt, vl_chat_processor, prompt_text, cot_prompt,
    num_generation=10, cfg_weight=5.0, temperature=1.0,
    image_token_num_per_image=576, img_size=384, patch_size=16,
):
    conversation = [
        {"role": "User", "content": cot_prompt.format(prompt_text)},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt='You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.',
    )
    prompt_inputs = vl_chat_processor.tokenizer(
        text=[sft_format], return_tensors="pt",
        padding=True, padding_side="right", add_special_tokens=True,
    )
    prompt_ids = prompt_inputs["input_ids"].repeat_interleave(num_generation, dim=0).to('cuda')
    prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(num_generation, dim=0).to('cuda')
    input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)

    completion_ids = mmgpt.language_model.generate(
        inputs_embeds=input_embeds, attention_mask=prompt_mask,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=512, do_sample=True, use_cache=True,
    )


    image_gen_prompt_list = []
    for i in range(completion_ids.shape[0]):
        answer = vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
        conv = [
            {"role": "User", "content": f"{prompt_text}. {answer}"},
            {"role": "Assistant", "content": ""},
        ]
        sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=vl_chat_processor.sft_format, system_prompt="",
        )
        image_gen_prompt_list.append(sft)

    prompt_inputs2 = vl_chat_processor.tokenizer(
        text=image_gen_prompt_list, return_tensors="pt",
        padding=True, padding_side="right", add_special_tokens=True,
    )
    prompt_ids2 = prompt_inputs2["input_ids"].to('cuda')
    attention_mask2 = prompt_inputs2["attention_mask"].to('cuda')

    image_start_token_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    prompt_ids2 = torch.cat([prompt_ids2, prompt_ids2.new_full((prompt_ids2.size(0), 1), image_start_token_id)], dim=1)
    attention_mask2 = torch.cat([attention_mask2, attention_mask2.new_ones((attention_mask2.size(0), 1))], dim=1)

    inputs_embeds2 = mmgpt.language_model.get_input_embeddings()(prompt_ids2)
    pad_input_embeds = mmgpt.language_model.get_input_embeddings()(
        prompt_ids2.new_full((1, 1), vl_chat_processor.pad_id)
    )

    generated_tokens = torch.zeros((num_generation, image_token_num_per_image), dtype=torch.int64).cuda()
    uncond_inputs_embeds = inputs_embeds2.clone()
    uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
    inputs_embeds_img = torch.repeat_interleave(inputs_embeds2, 2, dim=0)
    inputs_embeds_img[1::2] = uncond_inputs_embeds
    attention_mask_img = torch.repeat_interleave(attention_mask2, 2, dim=0)
    attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

    outputs = None
    for k in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds_img, use_cache=True,
            past_key_values=outputs.past_key_values if k != 0 else None,
            attention_mask=attention_mask_img,
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logits = logits[1::2] + cfg_weight * (logits[0::2] - logits[1::2])
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, k] = next_token.squeeze(dim=-1)
        next_token_rep = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_rep)
        inputs_embeds_img = img_embeds.unsqueeze(dim=1)
        attention_mask_img = torch.cat(
            [attention_mask_img, attention_mask_img.new_ones((attention_mask_img.shape[0], 1))], dim=1
        )

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[num_generation, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec


def run_category(mmgpt, vl_chat_processor, cot_prompt, cat_name, prompt_file, save_dir, args, overall_bar,
                 rank=0, world_size=1):
    os.makedirs(save_dir, exist_ok=True)
    with open(prompt_file, 'r') as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    # Shard prompts across ranks: rank r handles prompts[r::world_size]
    my_prompts = prompt_list[rank::world_size]

    if args.skip_existing:
        # 完整 = N 张图都在；partial 视为未完成，清掉残图重新生
        def is_complete(p):
            return all(
                os.path.exists(os.path.join(save_dir, f"{p}_{i:06d}.png"))
                for i in range(args.num_generation)
            )

        remaining = []
        partial = 0
        for p in my_prompts:
            if is_complete(p):
                continue
            # 不完整：若有任何残图，先清掉，避免下次再被误判 / 残留干扰
            had_any = False
            for i in range(args.num_generation):
                fpath = os.path.join(save_dir, f"{p}_{i:06d}.png")
                if os.path.exists(fpath):
                    os.remove(fpath)
                    had_any = True
            if had_any:
                partial += 1
            remaining.append(p)
        skipped = len(my_prompts) - len(remaining)
        if skipped:
            tqdm.write(f"  [rank{rank}] Skipping {skipped} complete prompts")
        if partial:
            tqdm.write(f"  [rank{rank}] Cleaned & re-queued {partial} partial prompts")
    else:
        remaining = my_prompts
        skipped = 0

    cat_bar = tqdm(
        remaining,
        desc=f"  [r{rank}] {cat_name:12s}",
        unit="prompt",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        position=1 + rank,
        disable=(rank != 0 and world_size > 1),  # 只 rank0 显示进度
    )

    success = skipped
    errors = 0
    t0 = time.time()

    for prompt_text in cat_bar:
        short = (prompt_text[:38] + "..") if len(prompt_text) > 40 else prompt_text
        cat_bar.set_postfix_str(f'"{short}"')
        try:
            images = generate_images_for_prompt(
                mmgpt, vl_chat_processor, prompt_text, cot_prompt,
                num_generation=args.num_generation, cfg_weight=args.cfg_weight,
            )
            for i, img_arr in enumerate(images):
                fname = f"{prompt_text}_{i:06d}.png"
                Image.fromarray(img_arr).save(os.path.join(save_dir, fname))
            success += 1
        except Exception as e:
            errors += 1
            tqdm.write(f"  [rank{rank}][ERROR] {prompt_text}: {e}")

        if rank == 0:
            overall_bar.update(world_size)  # 估算全局进度

    cat_bar.close()
    elapsed = time.time() - t0
    if rank == 0:
        tqdm.write(
            f"  ✅ {cat_name} rank0 DONE: {success}/{len(my_prompts)} prompts, "
            f"{success * args.num_generation} images, {errors} errors, {elapsed:.0f}s "
            f"(total prompts in category: {len(prompt_list)})"
        )
    return success, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_g8_1k6_full/checkpoint-800")
    parser.add_argument("--dataset_dir", type=str,
                        default="/mlx_devbox/users/xiezifan/playground/T2I-CompBench/examples/dataset")
    parser.add_argument("--save_root", type=str,
                        default="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results/finetuned")
    parser.add_argument("--reasoning_prompt_path", type=str,
                        default="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/data/prompt/reasoning_prompt.txt")
    parser.add_argument("--num_generation", type=int, default=10)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true", help="断点续跑，跳过已生成的 prompt")
    parser.add_argument("--categories", nargs="+",
                        default=["color", "shape", "texture", "spatial", "non_spatial", "complex"])
    args = parser.parse_args()

    # ── DDP init（torchrun 设置 RANK/WORLD_SIZE/LOCAL_RANK）────────────────────
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # 传 device_id，让 NCCL 在 init 时就在正确的卡上建 communicator
            # 否则首次 barrier 时会 fallback 到 cuda:0 申请 workspace，容易 OOM
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{local_rank}"),
            )
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(0)

    # 不同 rank 用不同 seed，避免每个 rank 采样路径完全一致
    seed_all(args.seed + rank)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Model : {args.model_path}")
        print(f"Output: {args.save_root}")
        print(f"World size: {world_size} (this rank={rank}, local_rank={local_rank})")
        print(f"{'='*60}")

    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    ).to(torch.bfloat16).cuda().eval()
    if rank == 0:
        print("✅ Model loaded.\n")

    with open(args.reasoning_prompt_path, 'r') as f:
        cot_prompt = f.read().strip()

    selected = [(n, f) for n, f in CATEGORIES if n in args.categories]

    total_prompts = 0
    already_done = 0   # 已完整生成的 prompt 数（N 张全在）；partial 不计
    for cat_name, val_file in selected:
        with open(os.path.join(args.dataset_dir, val_file)) as f:
            prompts_in_cat = [line.strip() for line in f if line.strip()]
        total_prompts += len(prompts_in_cat)
        save_dir = os.path.join(args.save_root, cat_name, "samples")
        if args.skip_existing and os.path.isdir(save_dir):
            for p in prompts_in_cat:
                if all(
                    os.path.exists(os.path.join(save_dir, f"{p}_{i:06d}.png"))
                    for i in range(args.num_generation)
                ):
                    already_done += 1

    if rank == 0:
        print(f"Categories   : {[n for n, _ in selected]}")
        print(f"Total prompts: {total_prompts} x {args.num_generation} = {total_prompts * args.num_generation} images")
        print(f"Already done : {already_done} prompts (will be skipped)")
        print(f"Sharded across {world_size} GPU(s): ~{total_prompts // world_size} prompts/GPU\n")

    overall_bar = tqdm(
        total=total_prompts,
        initial=already_done,
        desc="Overall   ",
        unit="prompt",
        position=0,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        disable=(rank != 0),
    )

    t_start = time.time()
    all_success, all_errors = 0, 0

    for i, (cat_name, val_file) in enumerate(selected):
        prompt_file = os.path.join(args.dataset_dir, val_file)
        save_dir = os.path.join(args.save_root, cat_name, "samples")
        if rank == 0:
            tqdm.write(f"\n[{i+1}/{len(selected)}] ── {cat_name.upper()} ── save → {save_dir}")
        s, e = run_category(
            vl_gpt, vl_chat_processor, cot_prompt,
            cat_name, prompt_file, save_dir, args, overall_bar,
            rank=rank, world_size=world_size,
        )
        all_success += s
        all_errors += e
        # 同步各 rank 完成同一 category 再进下一类，避免文件计数错位
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

    overall_bar.close()
    total_elapsed = time.time() - t_start

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"ALL DONE  {total_elapsed/3600:.1f}h total")
        print(f"  Success (rank0 only): {all_success} prompts / Errors: {all_errors}")
        print(f"{'='*60}")
        print("\nImage counts per category (across all ranks):")
        for cat_name, _ in selected:
            d = os.path.join(args.save_root, cat_name, "samples")
            n = len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.exists(d) else 0
            print(f"  {cat_name:12s}: {n:5d} images")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()