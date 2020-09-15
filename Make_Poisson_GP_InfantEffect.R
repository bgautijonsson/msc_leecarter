library(tidyverse)
library(rstan)

options(mc.cores = 4)

d <- read_csv("Data/combined_data.csv") %>% 
    filter(age < 100) %>% 
    group_by(age, year) %>% 
    summarise(deaths = sum(deaths),
              pop = sum(pop)) %>% 
    ungroup

N_obs <- nrow(d)
N_groups <- length(unique(d$age))
N_years <- length(unique(d$year))

deaths <- d$deaths
pop <- d$pop

age <- d$age + 1
year <- as.integer(d$year - min(d$year) + 1)

stan_data <- list(N_obs = N_obs, 
                  N_groups = N_groups,
                  N_years = N_years,
                  deaths = deaths, 
                  pop = pop, 
                  age = age,
                  year = year)

m <- stan(
    file = "Stan/LC_Pois_GaussianProcessPrior_InfantEffect.stan", 
    data  = stan_data,
    chains = 4, 
    iter = 2000, 
    warmup = 1000,
    control = list(max_treedepth = 15, adapt_delta = 0.9),
    seed = 1
)


write_rds(m, "Objects/PoisGPInfantEffect.rds")