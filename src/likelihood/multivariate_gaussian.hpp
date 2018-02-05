#ifndef MULTIVARIATE_GAUSSIAN_H
#define MULTIVARIATE_GAUSSIAN_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <chrono>
#include <random>
#include "../libs/eigenmvn/eigenmvn.h"

using namespace Eigen;
using namespace std;

class MVNGaussian{
    public:
        MVNGaussian();
        MVNGaussian(VectorXd _mean, MatrixXd _cov);
        MVNGaussian(VectorXd _mean, VectorXd _diag_cov);
        MVNGaussian(MatrixXd &data, bool  _diag = false);
        void generate(bool _cholesky = false);
        VectorXd getMean();
        MatrixXd getCov();
        VectorXd getdiagCov();
        MatrixXd getInvCov();
        VectorXd getdiagInvCov();
        void setMean(VectorXd &_mean);
        void setCov(MatrixXd &_cov);
        void setdiagCov(VectorXd &_diag_cov);
        VectorXd log_likelihood(MatrixXd data);
        VectorXd sample();
        MatrixXd sample(int n_samples);
    private:
        VectorXd mean;
        MatrixXd cov;
        VectorXd diag_cov;
        VectorXd diag_invcov;
        bool diag = false;
        MatrixXd invcov;
        mt19937 generator;
        int dim;
        EigenMultivariateNormal<double> normX_solver;
};


#endif // MULTIVARIATE_GAUSSIAN_H