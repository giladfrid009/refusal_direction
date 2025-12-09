from pipeline.model_utils.model_base import ModelBase


def construct_model_base(model_path: str) -> ModelBase:
    
    if "meta-llama/Llama-2-7b-chat-hf" == model_path:  
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    
    if "meta-llama/Llama-3.2-3B-Instruct" == model_path:  
        from pipeline.model_utils.llama32_model import Llama32Model
        return Llama32Model(model_path)
    
    if "microsoft/Phi-4-mini-instruct" == model_path: 
        from pipeline.model_utils.phi4_mini_model import Phi4MiniModel
        return Phi4MiniModel(model_path)
    
    if "google/gemma-2-2b-it" == model_path: 
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)        
    
    if "Qwen/Qwen3-4B-Instruct-2507" == model_path:  
        from pipeline.model_utils.qwen3_4b_model import Qwen3_2507_Model
        return Qwen3_2507_Model(model_path)
    
    if "ContinuousAT/Llama-2-7B-CAT" == model_path:  
        from pipeline.model_utils.llama2_cat_model import Llama2CATModel
        return Llama2CATModel(model_path)
    
    if "ContinuousAT/Phi-CAPO" == model_path: 
        from pipeline.model_utils.phi3_capo_model import Phi3CAPOModel
        return Phi3CAPOModel(model_path)

    if "cais/zephyr_7b_r2d2" == model_path:
        from pipeline.model_utils.zephyr_r2d2_model import ZephyrR2D2Model
        return ZephyrR2D2Model(model_path)

    if "GraySwanAI/Llama-3-8B-Instruct-RR" == model_path:
        from pipeline.model_utils.llama3_rr_model import Llama3RRModel
        return Llama3RRModel(model_path)
    
    if "LLM-LAT/robust-llama3-8b-instruct" == model_path:
        from pipeline.model_utils.llama3_robust_mode import Llama3RobustModel
        return Llama3RobustModel(model_path)

    raise ValueError(f"Unknown model family: {model_path}")

    if "qwen" in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    
    if "llama-3-" in model_path.lower():  # LLM-LAT/robust-llama3-8b-instruct, GraySwanAI/Llama-3-8B-Instruct-RR
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    
    if "gemma" in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    
    if "yi" in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    
    raise ValueError(f"Unknown model family: {model_path}")
