from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import sandbox.rocky.tf.core.layers as L
import lasagna.nonlinearities as NL
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc.instrument import stub, run_experiment_lite
from random_tabular_mdp import RandomTabularMDPEnv

stub(globals())

# parameters taken from paper
DISCOUNT = 0.99
POLICY_ITERS = 100 # up to 10000
GRU_UNITS = 256
BATCH_SIZE = 250000

# tabular mdp parameters from paper
N_STATES = 10
N_ACTIONS = 5
EPISODE_HORIZON = 10
N_EPISODES = 50 # they try several different numbers here 
EPISODE_LENGTH = 32 # they don't have a number for this parameter in the paper?
# mean parameters sampled from N(1,1)

tabular_env = RandomTabularMDPEnv(N_STATES, N_ACTIONS, N_EPISODES, EPISODE_LENGTH)
env = TfEnv(normalize(tabular_env))

policy = GaussianGRUPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(GRU_UNITS,),
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.softmax
    # gru_layer_cls=L.GRULayer,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=BATCH_SIZE,
    max_path_length=100,
    n_itr=POLICY_ITERS,
    discount=DISCOUNT,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
)