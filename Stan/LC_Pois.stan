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
}

parameters {
  real<lower = 0> sigma_state;
  vector[N_years - 1] z_kappa;
  // real kappa0;
  
  real theta;
  vector[N_groups] alpha;
  vector[N_groups - 1] beta_unconstrained;
}

transformed parameters {
  vector[N_groups] beta = simplex_transform(beta_unconstrained, N_groups);
}

model {
  vector[N_obs] mu_pred;
  
  vector[N_years] kappa;
  kappa[1] = 0;
  for (t in 2:N_years) kappa[t] = kappa[t - 1] + theta + z_kappa[t - 1] * sigma_state;
  
  mu_pred = alpha[age] + beta[age] .* kappa[year] + log_pop;
  
  target += exponential_lpdf(sigma_state | 1);
  target += std_normal_lpdf(z_kappa);
  target += normal_lpdf(theta | 0, 2);
  target += std_normal_lpdf(beta_unconstrained);
  target += poisson_log_lpmf(deaths | mu_pred);
}

generated quantities {
  real log_lik[N_obs];
  vector[N_years] kappa;
  kappa[1] = 0;
  
  for (t in 2:N_years) kappa[t] = kappa[t - 1] + theta + z_kappa[t - 1] * sigma_state;
  for (i in 1:N_obs) log_lik[i] = poisson_log_lpmf(deaths[i] | alpha[age[i]] + beta[age[i]] .* kappa[year[i]] + log_pop[i]);
}

