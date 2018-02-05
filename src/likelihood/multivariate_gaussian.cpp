#include "multivariate_gaussian.hpp"
MVNGaussian::MVNGaussian(){
    
}

MVNGaussian::MVNGaussian(VectorXd _mean, MatrixXd _cov){
    mean = _mean;
    cov = _cov;
    diag = false;
}

MVNGaussian::MVNGaussian(VectorXd _mean, VectorXd _diag_cov){
    mean = _mean;
    diag_cov = _diag_cov;
    if(diag_cov.all() >0 ){
        diag_invcov = 1.0/diag_cov.array();
    }
    else{
        diag_invcov = VectorXd::Ones(diag_cov.size());      
    }
    diag = true;
}

MVNGaussian::MVNGaussian(MatrixXd &data, bool  _diag){
    /* Getting mean for every column */
    mean = data.colwise().mean();
    diag = _diag;
    /* Covariance Matrix */
    if (diag){
        VectorXd var = (data.rowwise() - mean.transpose()).array().square().colwise().sum() / (data.rows());
        if(var.all() >0 ){
            diag_cov = var;
            diag_invcov = 1.0/var.array();
        } 
        else{
            cout << "Warning: Zeros in Variance vector, set Diagonal Unitary Covariance" << endl;
            diag_cov=VectorXd::Ones(data.cols());
            diag_invcov = diag_cov;      
        } 
    }
    else{
        MatrixXd centered = data.rowwise() - mean.transpose();
        cov = (centered.adjoint() * centered) / double(data.rows() - 1);
        invcov = cov.inverse();
    }
}

void MVNGaussian::generate(bool _cholesky){
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    if(!diag){
        normX_solver = EigenMultivariateNormal<double>(mean,cov, _cholesky, seed1);
    }
    else{
        generator.seed(seed1);
    }
}

VectorXd MVNGaussian::getMean(void){
    return mean;
}

MatrixXd MVNGaussian::getCov(void){
    return cov;
}

VectorXd MVNGaussian::getdiagCov(void){
    return diag_cov;
}

MatrixXd MVNGaussian::getInvCov(void){
    invcov = cov.inverse();
    return invcov;
}

VectorXd MVNGaussian::getdiagInvCov(void){
    return diag_invcov;
}

void MVNGaussian::setMean(VectorXd &_mean){
    mean = _mean;
}

void MVNGaussian::setCov(MatrixXd &_cov){
    cov = _cov;
}

void MVNGaussian::setdiagCov(VectorXd &_diag_cov){
    diag_cov = _diag_cov;
}

VectorXd MVNGaussian::sample(){
    dim = mean.size();
    VectorXd mvn = VectorXd::Ones(dim);
    if(!diag){
        mvn = normX_solver.samples(1);    
    }
    else{
        for (int i = 0; i < dim; ++i){
            normal_distribution<double> dnormal(mean(i),diag_cov(i));
            mvn(i) = dnormal(generator);
        }
    }
    return mvn;
}

MatrixXd MVNGaussian::sample(int n_samples){
    dim = mean.size();
    MatrixXd mvn = MatrixXd::Ones(n_samples, dim);
    if(!diag){
        mvn = normX_solver.samples(n_samples);
    }
    else{
        for (int j = 0; j < n_samples; ++j){
            for (int i = 0; i < dim; ++i){
                normal_distribution<double> dnormal(mean(i),diag_cov(i));
                mvn(j, i) = dnormal(generator);
            }
        }
    }
    return mvn;
}

VectorXd MVNGaussian::log_likelihood(MatrixXd data){
    double rows = data.rows();
    double cols = data.cols();
    VectorXd loglike = VectorXd::Zero(rows);
    /* Getting inverse matrix for 'cov' with Cholesky */
    LLT<MatrixXd> chol(cov);
    MatrixXd L = chol.matrixL();
    MatrixXd cov_inverse = L.adjoint().inverse() * L.inverse();
    double logdet=log(cov.determinant());
     for(unsigned i=0;i<rows;i++){
        VectorXd tmp1 = data.row(i);
        tmp1 -= mean;
        MatrixXd tmp2 = tmp1.transpose() * cov_inverse;
        tmp2 = tmp2 * tmp1;
        loglike(i) = -0.5 * tmp2(0,0) - (cols/2) * log(2*M_PI) -(0.5) * logdet;
    }
    //cout << loglike << endl;
    return loglike;
}
