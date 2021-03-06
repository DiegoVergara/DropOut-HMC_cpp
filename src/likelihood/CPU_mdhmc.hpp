//Author: Diego Vergara
#ifndef Mask_CPU_HAMILTONIAN_MC_H
#define Mask_CPU_HAMILTONIAN_MC_H
#include "CPU_softmax_regression.hpp"
#include "hmc.hpp"

class Mask_CPU_Hamiltonian_MC : public Hamiltonian_MC
{
public:
	void init( MatrixXd &_X, VectorXd &_Y, double _lambda = 1.0, int _warmup_iterations = 100, int _iterations = 1000, int _minibatch = 1000, double _step_size = 0.01, int _num_step = 100,  double _mask_rate = 0.5, bool _normalization =true, bool _standarization=true, int _samples = 1000, string _path="", double _path_lenght = 0.0);
	void run(bool warmup_flag = false, bool for_predict = false, double mom = 0.99);
	VectorXd predict(MatrixXd &_X_test, int psamples = 1, bool simulation = false, bool data_processing = true);
	MatrixXd get_maskMatrix();
	void set_maskMatrix(MatrixXd &_mask_matrix);
	void getModel(MatrixXd& weights, MatrixXd& mask_matrix, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin);
	void loadModel(MatrixXd weights, MatrixXd mask_matrix, VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, bool _normalization = true, bool _standarization =true);
	VectorXd gradient(VectorXd &W,VectorXd &mask, int n_iter);
	double logPosterior(VectorXd &W, VectorXd &mask, int n_iter);
	void setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing = true);
	void saveModel(string name);
	void getMaskStats(double& _mask_mean, double& _mask_std);
protected:
	MatrixXd mask_matrix;
	double mask_rate;
 	CPU_SoftmaxRegression softmax_regression;
 	bool normalization, standarization;
 	int n_classes,cols;
 	double mask_mean, mask_std;
};

#endif // Mask_CPU_HAMILTONIAN_MC_H