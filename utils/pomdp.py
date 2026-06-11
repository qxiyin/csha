import numpy as np
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading


class ContinuousPOMDP:

    def __init__(self, num_clients, device='cpu'):
        self.num_clients = num_clients
        self.device = device

        self.gamma = 0.95
        self.Ca = 0.8
        self.CL = 0.3
        self.CR = 0.1
        self.Theta = 5
        self.fourier_coeffs_c = np.random.normal(0, 0.1, self.Theta)
        self.fourier_coeffs_d = np.random.normal(0, 0.1, self.Theta)
        self.num_particles = 30
        self.particles = np.random.uniform(0, 1, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.mu = 1.5
        self.population_size = 20
        self.max_generations = 30
        self.M = 40
        self.k = 12
        self.pmin = 0.1
        self.pmax = 0.9
        self.beta1 = 0.1
        self.beta2 = 0.1
        self.beta3 = 0.1
        self.tau_upper_max = 5.0
        self.tau_upper_unit = 0.1
        self.g1 = 8
        self.g2 = 15
        self.alpha1 = 0.01
        self.beta4 = 0.1
        self.beta5 = 0.05
        self.mutation_factors = {}
        self.tau_g = 1.0
        self.current_belief_state = np.ones(self.M) / self.M
        self.state_history = []
        self.observation_history = []
        self.use_parallel_fitness = True
        self.num_workers = min(2, mp.cpu_count())
        self.convergence_threshold = 1e-4
        self.convergence_patience = 3

        logging.info(f"[POMDP] Initialized with {num_clients} clients (theory-compliant optimization)")

    def fourier_belief_approximation(self, particles, weights):
        weights = weights / (np.sum(weights) + 1e-8)

        for i in range(self.Theta):
            angle = 2 * np.pi * i * particles
            self.fourier_coeffs_c[i] = np.sum(weights * np.sin(angle))
            self.fourier_coeffs_d[i] = np.sum(weights * np.cos(angle))

        total_weight = np.sum(np.abs(self.fourier_coeffs_c)) + np.sum(np.abs(self.fourier_coeffs_d))
        if total_weight > 1e-8:
            self.fourier_coeffs_c /= total_weight
            self.fourier_coeffs_d /= total_weight

    def compute_ess(self, weights):
        return 1.0 / (np.sum(weights ** 2) + 1e-8)

    def kld_sampling(self, k, zeta=0.01, eta=0.01):
        z_quantile = 1.96
        if k <= 1:
            return max(10, self.num_particles // 2)

        numerator = k - 1
        denominator = 2 * zeta
        term1 = 1 - 2 / (9 * (k - 1))
        term2 = np.sqrt(2 / (9 * (k - 1))) * z_quantile
        N = numerator / denominator * (term1 + term2) ** 3
        return max(int(N), 10)

    def adaptive_particle_filtering(self, observation, action):
        noise_std = 0.05
        for i in range(len(self.particles)):
            noise = np.random.normal(0, noise_std)
            if action == 1:
                self.particles[i] = max(0, min(1, self.particles[i] * 0.8 + noise))
            else:
                self.particles[i] = max(0, min(1, self.particles[i] * 1.1 + noise))

        for i in range(len(self.particles)):
            obs_prob = np.exp(-0.5 * (observation - self.particles[i]) ** 2 / 0.01)
            self.particles[i] = obs_prob

        total_weight = np.sum(self.weights)
        if total_weight > 1e-8:
            self.weights /= total_weight
        else:
            self.weights = np.ones(len(self.particles)) / len(self.particles)

        ess = self.compute_ess(self.weights)
        if len(self.particles) / ess > self.mu:
            new_size = self.kld_sampling(len(np.unique(self.particles)))
            if new_size != len(self.particles):
                indices = np.random.choice(len(self.particles), size=new_size, p=self.weights)
                self.particles = self.particles[indices]
                self.weights = np.ones(len(self.particles)) / len(self.particles)

        self.fourier_belief_approximation(self.particles, self.weights)

    def optimized_evaluate_fitness(self, chromosome, observation):
        if np.sum(chromosome) == 0 or np.sum(chromosome) > self.k:
            return -np.inf

        selected_states = np.where(chromosome == 1)[0] / self.M
        total_reward = 0

        for state in selected_states:
            state_prob = 0
            for i in range(self.Theta):
                state_prob += self.fourier_coeffs_c[i] * np.sin(2 * np.pi * i * state)
                state_prob += self.fourier_coeffs_d[i] * np.cos(2 * np.pi * i * state)
            state_prob = max(0, state_prob)

            reward_monitor = -self.Ca * state
            reward_check = -self.CL - self.CR * state

            max_reward = max(reward_monitor, reward_check)
            total_reward += state_prob * max_reward

        size_penalty = -0.01 * np.sum(chromosome)
        return total_reward + size_penalty

    def parallel_evaluate_fitness(self, population, observation):
        if not self.use_parallel_fitness or len(population) < 4:
            return [self.optimized_evaluate_fitness(chromosome, observation) for chromosome in population]

        try:
            fitness_args = [(chromosome, observation, self._get_fitness_params())
                            for chromosome in population]

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                fitness_scores = list(executor.map(self._evaluate_fitness_worker, fitness_args))
            return fitness_scores
        except Exception as e:
            logging.warning(f"[POMDP] Parallel fitness evaluation failed: {e}")
            return [self.optimized_evaluate_fitness(chromosome, observation) for chromosome in population]

    def evolutionary_state_subset_optimization(self, observation):
        population = []
        for u in range(self.population_size):
            chromosome = np.zeros(self.M, dtype=int)
            k_prime = max(1, round(self.k * 0.3))
            selected_indices = np.random.choice(self.M, size=min(k_prime, self.M), replace=False)
            chromosome[selected_indices] = 1
            population.append(chromosome)
            self.mutation_factors[u] = np.random.uniform(0.1, 1.0)

        best_fitness = -np.inf
        best_solution = None
        no_improvement_count = 0

        for generation in range(self.max_generations):
            fitness_scores = self.parallel_evaluate_fitness(population, observation)

            for i, fitness in enumerate(fitness_scores):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = population[i].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            if no_improvement_count >= self.convergence_patience:
                logging.debug(f"[POMDP] Early stopping at generation {generation}")
                break

            self._update_lmp_parameters(generation, fitness_scores)

            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = max(1, self.population_size // 4)
            new_population = [population[i] for i in sorted_indices[:elite_size]]

            offspring_count = 0
            while len(new_population) < self.population_size:
                parent1_idx = sorted_indices[np.random.randint(elite_size)]
                parent2_idx = sorted_indices[np.random.randint(elite_size)]
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                child = self.adaptive_sparsity_preserved_crossover(
                    parent1, parent2,
                    fitness_scores[parent1_idx], fitness_scores[parent2_idx],
                    min(fitness_scores), max(fitness_scores)
                )

                child = self.lmp_based_sparsity_preserved_mutation(child, generation, offspring_count)

                new_population.append(child)
                offspring_count += 1

            population = new_population

        return best_solution if best_solution is not None else population[0]

    def adaptive_sparsity_preserved_crossover(self, parent1, parent2, f1, f2, fmin, fmax):

        child = np.zeros_like(parent1, dtype=int)

        if fmax - fmin > 1e-8:
            pc = self.pmin + (self.pmax - self.pmin) * (1 - abs(f1 - f2) / (fmax - fmin))
        else:
            pc = self.pmax

        xor_result = np.logical_xor(parent1, parent2).astype(int)
        h = np.sum(xor_result)

        if h == 0:
            return parent1.copy()

        a_star = np.sum(parent2 * xor_result)
        b_star = np.sum(parent1 * xor_result)

        if a_star > 0 and b_star > 0:
            kappa = h / (2 * a_star) + h / (2 * b_star)

            p01 = (pc * h) / (2 * a_star * kappa)
            p10 = (pc * h) / (2 * b_star * kappa)

            p01 = min(1.0, max(0.0, p01))
            p10 = min(1.0, max(0.0, p10))
        else:
            p01 = p10 = 0.0

        for i in range(len(parent1)):
            if parent1[i] == 0 and parent2[i] == 1:
                if np.random.random() < p01:
                    child[i] = 1
                else:
                    child[i] = 0
            elif parent1[i] == 1 and parent2[i] == 0:
                if np.random.random() < p10:
                    child[i] = 0
                else:
                    child[i] = 1
            else:
                child[i] = parent1[i]

        return child

    def lmp_based_sparsity_preserved_mutation(self, chromosome, generation, chromosome_idx):

        mutated = chromosome.copy().astype(int)
        M = len(chromosome)

        if generation <= self.g1:
            pm = self._normalize_mutation_factor(self.mutation_factors.get(chromosome_idx, 1.0))
        elif generation <= self.g2:
            pm = self.alpha1 * generation + self.beta4
        else:
            pm = self.beta5

        pm = min(1.0, max(0.0, pm))

        a = np.sum(chromosome == 0)
        b = np.sum(chromosome == 1)

        if a == 0 or b == 0:
            return mutated

        nu = M / (2 * a) + M / (2 * b)

        p0 = (pm * M) / (2 * a * nu)
        p1 = (pm * M) / (2 * b * nu)

        p0 = min(1.0, max(0.0, p0))
        p1 = min(1.0, max(0.0, p1))

        for i in range(M):
            if chromosome[i] == 0:
                if np.random.random() < p0:
                    mutated[i] = 1
            else:
                if np.random.random() < p1:
                    mutated[i] = 0

        if np.sum(mutated) > self.k:
            ones_indices = np.where(mutated == 1)[0]
            excess = len(ones_indices) - self.k
            if excess > 0:
                remove_indices = np.random.choice(ones_indices, size=excess, replace=False)
                mutated[remove_indices] = 0

        return mutated

    def _update_lmp_parameters(self, generation, fitness_scores):
        if len(fitness_scores) == 0:
            return

        U = len(fitness_scores)
        sum_fitness = sum(fitness_scores)
        max_fitness = max(fitness_scores)

        if max_fitness <= -1e6:
            return

        if max_fitness > 0:
            tau_upper_g = self.beta3 * sum_fitness / max_fitness * (
                    self.tau_upper_max - generation * self.tau_upper_unit
            )
        else:
            tau_upper_g = self.tau_upper_max - generation * self.tau_upper_unit

        tau_upper_g = max(0.1, tau_upper_g)

        if tau_upper_g > 0:
            dtau_dg = self.beta2 * self.tau_g * (1 - sum(self.mutation_factors.values()) / tau_upper_g)
            self.tau_g = max(0.1, self.tau_g + dtau_dg)

        for u in range(min(U, len(self.mutation_factors))):
            if u in self.mutation_factors and self.tau_g > 0:
                dru_dg = self.beta1 * self.mutation_factors[u] * (
                        np.log(max(self.tau_g, 1e-8)) - np.log(max(self.mutation_factors[u], 1e-8))
                )
                self.mutation_factors[u] = max(0.01, min(10.0, self.mutation_factors[u] + dru_dg))

    def _normalize_mutation_factor(self, r_u_g):
        return min(1.0, max(0.0, r_u_g / 10.0))

    def compute_state_probability(self, state):
        prob = 0
        for i in range(self.Theta):
            prob += self.fourier_coeffs_c[i] * np.sin(2 * np.pi * i * state)
            prob += self.fourier_coeffs_d[i] * np.cos(2 * np.pi * i * state)
        return max(0, prob)

    def solve_pomdp(self, observation):
        self.adaptive_particle_filtering(observation, 0)

        optimal_subset = self.evolutionary_state_subset_optimization(observation)

        reward_monitor = self.compute_expected_reward(optimal_subset, observation, action=0)
        reward_check = self.compute_expected_reward(optimal_subset, observation, action=1)

        logging.info(f"[POMDP] Rewards - Monitor: {reward_monitor:.6f}, Check: {reward_check:.6f}")

        return 1 if reward_check > reward_monitor else 0

    def compute_expected_reward(self, state_subset, observation, action):
        selected_states = np.where(state_subset == 1)[0] / self.M
        total_reward = 0

        for state in selected_states:
            state_prob = self.compute_state_probability(state)

            if action == 0:
                reward = -self.Ca * state
            else:
                reward = -self.CL - self.CR * state

            total_reward += state_prob * reward

        return total_reward

    def _get_fitness_params(self):
        return {
            'k': self.k,
            'M': self.M,
            'Ca': self.Ca,
            'CL': self.CL,
            'CR': self.CR,
            'Theta': self.Theta,
            'fourier_coeffs_c': self.fourier_coeffs_c.copy(),
            'fourier_coeffs_d': self.fourier_coeffs_d.copy()
        }

    @staticmethod
    def _evaluate_fitness_worker(args):
        chromosome, observation, params = args

        if np.sum(chromosome) == 0 or np.sum(chromosome) > params['k']:
            return -np.inf

        selected_states = np.where(chromosome == 1)[0] / params['M']
        total_reward = 0

        for state in selected_states:
            state_prob = 0
            for i in range(params['Theta']):
                state_prob += params['fourier_coeffs_c'][i] * np.sin(2 * np.pi * i * state)
                state_prob += params['fourier_coeffs_d'][i] * np.cos(2 * np.pi * i * state)
            state_prob = max(0, state_prob)

            reward_monitor = -params['Ca'] * state
            reward_check = -params['CL'] - params['CR'] * state

            max_reward = max(reward_monitor, reward_check)
            total_reward += state_prob * max_reward

        size_penalty = -0.01 * np.sum(chromosome)
        return total_reward + size_penalty