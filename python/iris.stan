data {
    int<lower=0> n;
    int<lower=0,upper=1> species[n];
    real sepal_length[n];
    real sepal_width[n];
}
transformed data {}

parameters {
    real beta_0;
    real beta_1;
    real alpha;
}
transformed parameters {}

model {
    beta_0 ~ normal(0, 10);
    beta_1 ~ normal(0, 10);
    alpha ~ normal(0, 10);
    for(i in 1:n)
        species[i] ~ bernoulli(inv_logit(alpha+beta_0*sepal_length[i]+beta_1*sepal_width[i]));
}

