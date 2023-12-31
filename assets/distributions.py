from abc import ABC, abstractmethod
from typing import Optional, Tuple, TypeVar, Union
import torch as T
from torch import nn
from torch.distributions import Normal, Categorical

SelfDiagGaussianDistribution = TypeVar(
    "SelfDiagGaussianDistribution", bound="DiagGaussianDistribution"
)
SelfDistribution = TypeVar("SelfDistribution", bound="Distribution")
SelfStateDependentNoiseDistribution = TypeVar(
    "SelfStateDependentNoiseDistribution", bound="StateDependentNoiseDistribution"
)
SelfSquashedDiagGaussianDistribution = TypeVar(
    "SelfSquashedDiagGaussianDistribution", bound="SquashedDiagGaussianDistribution"
)
SelfCategoricalDistribution = TypeVar(
    "SelfCategoricalDistribution", bound="CategoricalDistribution"
)


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(
        self, *args, **kwargs
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self: SelfDistribution, *args, **kwargs) -> SelfDistribution:
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: T.Tensor) -> T.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[T.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> T.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> T.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> T.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> T.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[T.Tensor, T.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


def sum_independent_dims(tensor: T.Tensor) -> T.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(
            T.ones(self.action_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def proba_distribution(
        self: SelfDiagGaussianDistribution, mean_actions: T.Tensor, log_std: T.Tensor
    ) -> SelfDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = T.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: T.Tensor) -> T.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> T.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> T.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> T.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self, mean_actions: T.Tensor, log_std: T.Tensor, deterministic: bool = False
    ) -> T.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: T.Tensor, log_std: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(
        self: SelfSquashedDiagGaussianDistribution,
        mean_actions: T.Tensor,
        log_std: T.Tensor,
    ) -> SelfSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(
        self, actions: T.Tensor, gaussian_actions: Optional[T.Tensor] = None
    ) -> T.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= T.sum(T.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[T.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> T.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return T.tanh(self.gaussian_actions)

    def mode(self) -> T.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return T.tanh(self.gaussian_actions)

    def log_prob_from_params(
        self, mean_actions: T.Tensor, log_std: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self: SelfCategoricalDistribution, action_logits: T.Tensor
    ) -> SelfCategoricalDistribution:
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: T.Tensor) -> T.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> T.Tensor:
        return self.distribution.entropy()

    def sample(self) -> T.Tensor:
        return self.distribution.sample()

    def mode(self) -> T.Tensor:
        return T.argmax(self.distribution.probs, dim=1)

    def actions_from_params(
        self, action_logits: T.Tensor, deterministic: bool = False
    ) -> T.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, action_logits: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719
    Github: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: T.Tensor) -> T.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = T.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (T.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = T.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return T.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, log_std: T.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(T.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = -2.0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        # Network for the deterministic action, it represents the mean of the distribution
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = (
            T.ones(self.latent_sde_dim, self.action_dim)
            if self.full_std
            else T.ones(self.latent_sde_dim, 1)
        )
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
        self: SelfStateDependentNoiseDistribution,
        mean_actions: T.Tensor,
        log_std: T.Tensor,
        latent_sde: T.Tensor,
    ) -> SelfStateDependentNoiseDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = T.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, T.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: T.Tensor) -> T.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= T.sum(
                self.bijector.log_prob_correction(gaussian_actions), dim=1
            )
        return log_prob

    def entropy(self) -> Optional[T.Tensor]:
        if self.bijector is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> T.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def mode(self) -> T.Tensor:
        actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def get_noise(self, latent_sde: T.Tensor) -> T.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return T.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = T.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def actions_from_params(
        self,
        mean_actions: T.Tensor,
        log_std: T.Tensor,
        latent_sde: T.Tensor,
        deterministic: bool = False,
    ) -> T.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: T.Tensor, log_std: T.Tensor, latent_sde: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: T.Tensor) -> T.Tensor:
        return T.tanh(x)

    @staticmethod
    def atanh(x: T.Tensor) -> T.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: T.Tensor) -> T.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = T.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: T.Tensor) -> T.Tensor:
        # Squash correction (from original SAC implementation)
        return T.log(1.0 - T.tanh(x) ** 2 + self.epsilon)
