import time
import jax
import jax.numpy as jnp
from istratde.util import Problem, StdSOMonitor, StdWorkflow, dataclass
from istratde.algorithms.jax import IStratDE
@dataclass
class Sphere(Problem):
    def __init__(self): pass
    def evaluate(self, state, X): return jax.vmap(lambda x: jnp.sum(x*x))(X), state
print("devices", jax.devices())
D=10; POP_SIZE=4096; STEPS=30
algorithm=IStratDE(lb=jnp.full((D,),-100.0), ub=jnp.full((D,),100.0), pop_size=POP_SIZE)
monitor=StdSOMonitor(record_fit_history=False)
workflow=StdWorkflow(algorithm=algorithm, problem=Sphere(), monitor=monitor)
state=workflow.init(jax.random.PRNGKey(42))
t0=time.perf_counter()
for _ in range(STEPS): state=workflow.step(state)
jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)
print("elapsed_s", round(time.perf_counter()-t0,3), "best", monitor.get_best_fitness())
