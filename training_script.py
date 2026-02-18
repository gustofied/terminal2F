from verifiers_rl.rl.trainer import RLTrainer, RLConfig, get_model_and_tokenizer
import verifiers as vf
import wandb


def main():
    wandb.login()

    model_name = "willcb/Qwen3-0.6B"
    vf_env = vf.load_environment("alphabet-sort")

    # RLConfig — NOT the same as the tutorial's grpo_defaults()
    # Key differences from blog tutorial:
    #   - use_lora defaults to True (tutorial assumes False)
    #   - use_liger defaults to True (requires flash-attn)
    #   - batch_size controls total rollouts, not per_device_train_batch_size
    #   - gradient_accumulation_steps is hardcoded to 1 in __post_init__
    #   - rollouts_per_example replaces num_generations
    #   - micro_batch_size replaces per_device_train_batch_size
    args = RLConfig(
        run_name="alphasort-grpo-qwen-3",
        # Model loading — use_liger=True is fine with flash-attn properly installed
        use_liger=True,
        # LoRA — disable for full fine-tuning (tutorial default)
        use_lora=False,
        # Batch config
        batch_size=64,          # total rollouts per batch (default 512, too big)
        micro_batch_size=4,     # per GPU per step
        rollouts_per_example=8, # GRPO group size (default 16)
        max_seq_len=2048,
        # Training
        max_steps=1000,
        max_concurrent=1024,
        generation_timeout=300.0,
        # Monitoring
        logging_steps=1,
        report_to="wandb",
        # Saving
        output_dir="./mymodel",
        overwrite_output_dir=True,
        hub_model_id="gustofied/Qwen3-0.6B-alphabet-sort-grpo",
        hub_strategy="every_save",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        push_to_hub=True,
    )

    # Load model — use_liger=True uses AutoLigerKernelForCausalLM (flash-attn installed)
    model, tokenizer = get_model_and_tokenizer(model_name, use_liger=True)

    # RLTrainer — NOT GRPOTrainer (wrapper has broken positional arg order)
    # Signature: RLTrainer(model, env, args, processing_class=None)
    trainer = RLTrainer(
        model=model,
        env=vf_env,
        args=args,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
