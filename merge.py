from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir="/home/zss/Social_Behavior_Simulation/checkpoints/grpo_checkpoints/iter_27.00/global_step_26/actor",
            target_dir="/home/zss/Social_Behavior_Simulation/checkpoints/grpo_checkpoints/ite27_globalstep26",
            hf_model_config_path="/home/zss/Social_Behavior_Simulation/checkpoints/grpo_checkpoints/iter_27.00/global_step_26/actor/huggingface"
)
merger = FSDPModelMerger(config)
merger.merge_and_save()