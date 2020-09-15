functions {
  vector simplex_transform(vector z, int K) {
    vector[K - 1] x;
    vector[K] out;
    
    x[1] = inv_logit(z[1] - log(K - 1)); 
    out[1] = x[1];
    
    for (k in 2:(K - 1)) {
      x[k] = inv_logit(z[k] - log(K - k)); 
      out[k] = (1 - sum(out[1:(k - 1)])) * x[k];
    }
    
    out[K] = 1 - sum(out[1:(K - 1)]);
    return(out);
  }
  

  
  vector gp_to_pars(real mu, real sigma, real[] x, real eta, real rho, vector z, int N, real infant_effect) {
    matrix[N, N] K = cov_exp_quad(x, eta, rho) + diag_matrix(rep_vector(square(sigma), N));
    matrix[N, N] LK = cholesky_decompose(K);
    vector[N] out = rep_vector(mu, N) + LK * z;
    out[1] += infant_effect;
    return(out);
  }
}


data {
  int<lower = 0> N_obs;
  int<lower = 0> N_groups;
  int<lower = 0> N_years;
  vector[N_obs] pop;
  int deaths[N_obs];
  int age[N_obs];
  int year[N_obs];
}

transformed data {
  vector[N_obs] log_pop = log(pop);
  real ages[N_groups];
  real<lower = 0> N_years_numeric = N_years;
  for (i in 1:N_groups) ages[i] = i;
}

parameters {
  real<lower = 0> sigma_state;
  vector[N_years - 1] z_kappa;
  
  real theta;
  
  real mu_alpha;
  real alpha_infant;
  real<lower = 0> sigma_alpha;
  vector[N_groups] z_alpha;
  
  real mu_beta;
  real beta_infant;
  real<lower = 0> sigma_beta;
  vector[N_groups - 1] z_beta;
  
  real<lower = 0> rho_alpha;
  real<lower = 0> rho_beta;
  
  real<lower = 0> eta_alpha;
  real<lower = 0> eta_beta;
  
  real<lower = 0> phi_inv_sqrt;
  
  
}
// gp_to_pars(real mu, real sigma, real[] x, real alpha, real rho, vector z, int N)
transformed parameters {
  vector[N_groups - 1] beta_unconstrained = gp_to_pars(mu_beta, sigma_beta, ages[1:(N_groups - 1)], eta_beta, rho_beta, z_beta, N_groups - 1, beta_infant);
  vector[N_groups] beta = simplex_transform(beta_unconstrained, N_groups);
  vector[N_groups] alpha = gp_to_pars(mu_alpha, sigma_alpha, ages, eta_alpha, rho_alpha, z_alpha, N_groups, alpha_infant);
  real<lower = 0> phi_inv = square(phi_inv_sqrt);
  real<lower = 0> phi = inv(phi_inv);
  
  
}

model {
  vector[N_obs] mu_pred;
  vector[N_years] kappa;
  kappa[1] = 0;
  for (t in 2:N_years) kappa[t] = kappa[t - 1] + theta + z_kappa[t - 1] * sigma_state;
  mu_pred = alpha[age] + beta[age] .* kappa[year] + log_pop;
  
  target += inv_gamma_lpdf(rho_alpha | 1.7, 8.5);
  target += inv_gamma_lpdf(rho_beta | 1.7, 8.5);
  target += std_normal_lpdf(eta_alpha);
  target += std_normal_lpdf(eta_beta);
  target += std_normal_lpdf(z_beta);
  target += std_normal_lpdf(z_alpha);
  target += exponential_lpdf(sigma_alpha | 1);
  target += exponential_lpdf(sigma_beta | 1);
  target += exponential_lpdf(sigma_state | 1);
  target += normal_lpdf(theta | 0, 2);
  target += std_normal_lpdf(z_kappa);
  target += std_normal_lpdf(mu_beta);
  target += normal_lpdf(mu_alpha | -5, 2);
  target += std_normal_lpdf(alpha_infant);
  target += std_normal_lpdf(beta_infant);
  target += exponential_lpdf(phi_inv_sqrt | 1);
  target += neg_binomial_2_log_lpmf(deaths | mu_pred, phi);
} 

generated quantities {
  real log_lik[N_obs];
  vector[N_years] kappa;
  kappa[1] = 0;
  for (t in 2:N_years) kappa[t] = kappa[t - 1] + theta + z_kappa[t - 1] * sigma_state;
  for (i in 1:N_obs) log_lik[i] = neg_binomial_2_log_lpmf(deaths[i] | alpha[age[i]] + beta[age[i]] .* kappa[year[i]] + log_pop[i], phi);
}

