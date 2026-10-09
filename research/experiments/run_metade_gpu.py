import time
import jax
import jax.numpy as jnp
from metade.util import StdSOMonitor, StdWorkflow
from metade.algorithms.jax import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE
from metade.util import Problem, dataclass
@dataclass
class Sphere(Problem):
    def __init__(self): pass
    def evaluate(self, state, X): return jax.vmap(lambda x: jnp.sum(x*x))(X), state
print("devices", jax.devices())
D=10; BATCH_SIZE=16; NUM_RUNS=1; OUTER_STEPS=8; BASE_POP=64; BASE_STEPS=20
tiny=1e-5
param_lb=jnp.array([0,0,0,0,1,0]); param_ub=jnp.array([1,1,4-tiny,4-tiny,5-tiny,3-tiny])
evolver=DE(lb=param_lb, ub=param_ub, pop_size=BATCH_SIZE, base_vector="rand", differential_weight=0.5, cross_probability=0.9)
BatchDE=create_batch_algorithm(ParamDE, BATCH_SIZE, NUM_RUNS)
batch_de=BatchDE(lb=jnp.full((D,),-100), ub=jnp.full((D,),100), pop_size=BASE_POP)
meta_problem=MetaDE(batch_de, Sphere(), batch_size=BATCH_SIZE, num_runs=NUM_RUNS, base_alg_steps=BASE_STEPS)
monitor=StdSOMonitor(record_fit_history=False)
workflow=StdWorkflow(algorithm=evolver, problem=meta_problem, pop_transform=decoder_de, monitor=monitor, record_pop=True)
state=workflow.init(jax.random.PRNGKey(42)); t0=time.perf_counter()
for i in range(OUTER_STEPS): state=state.update_child("problem", {"power_up": 1 if i==OUTER_STEPS-1 else 0}); state=workflow.step(state)
jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x,"block_until_ready") else x, state)
print("elapsed_s", round(time.perf_counter()-t0,3), "best", monitor.get_best_fitness())
