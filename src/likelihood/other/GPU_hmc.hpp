//Author: Diego Vergara
#ifndef GPU_HAMILTONIAN_MC_H
#define GPU_HAMILTONIAN_MC_H
#include "GPU_logistic_regression.hpp"
#include "hmc.hpp"


class GPU_Hamiltonian_MC : public Hamiltonian_MC
{
public:
	void init( MatrixXd &_X, VectorXd &_Y, double _lambda = 1.0, int _warmup_iterations = 100, int _iterations = 1000, double _step_size = 0.01, int _num_step = 100, bool _normalization =true, bool _standarization=true, double _path_lenght = 0.0);
	void run(bool warmup_flag = false, bool for_predict = false);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 1, bool simulation = false);
	void getModel(VectorXd& weights, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin, double& bias);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	VectorXd gradient(VectorXd &W);
	double logPosterior(VectorXd &W, bool precompute = true);
	void setData(MatrixXd &_X,VectorXd &_Y);
protected:
	GPU_LogisticRegression logistic_regression;
	bool normalization, standarization;
};

#endif // GPU_HAMILTONIAN_MC_H