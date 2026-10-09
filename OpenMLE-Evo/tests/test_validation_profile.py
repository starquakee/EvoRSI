from pathlib import Path
from hydra import initialize_config_dir, compose
from research.search.validation_config import validate_inner_evo_config


def test_composed_profile_matches_real_backend_and_solver_constraints():
    config_dir=Path(__file__).resolve().parents[1]/'tts_search/configs'
    with initialize_config_dir(version_base=None,config_dir=str(config_dir)):
        cfg=compose(config_name='experiment/trustworthy_validation')
    solver=cfg.search.runner.solver
    model=cfg.litellm.model_list[0].litellm_params
    validate_inner_evo_config({
        'num_generations':solver.num_generations,
        'individuals_per_generation':solver.individuals_per_generation,
        'step_limit':min(cfg.max_steps,solver.step_limit),
        'num_islands':solver.num_islands,
        'num_generations_till_crossover':solver.num_generations_till_crossover,
        'llm_concurrency':cfg.llm_concurrency,
        'sandbox_concurrency':cfg.sandbox.concurrency,
        'experience_enabled':solver.experience.enabled,
        'prompt_memory_enabled':solver.experience.prompt_memory.enabled,
        'max_output_tokens':model.max_tokens,
        'stream':model.stream,
    })
    assert model.model=='openai/k3'
    assert model.api_key=='runtime'
    assert model.num_retries==model.max_retries==0
    assert solver.experience.parent_selection.enabled is True
