//Author: Diego Vergara
#include "CPU_dhmc.hpp"

void Mask_CPU_Hamiltonian_MC::init(MatrixXd &_X, VectorXd &_Y, double _lambda, int _warmup_iterations, int _iterations, int _minibatch, double _step_size, int _num_step, double _mask_rate, bool _normalization, bool _standarization, int _samples, string _path,double _path_length){
	this->lambda=_lambda;
	this->step_size = _step_size;
	this->num_step = _num_step;
	this->path_length = _path_length;
	if (this->path_length > 0.0) this->num_step = int(this->path_length/this->step_size);
	this->warmup_iterations = _warmup_iterations;
	this->X_train = &_X;
 	this->Y_train = &_Y;
	this->rows = _X.rows();
	this->dim = _X.cols()+1; // + bias
	this->normalization = _normalization;
    this->standarization = _standarization;
    this->logistic_regression.init(_X, _Y, this->lambda, this->normalization, this->standarization, true);
    this->init_hmc = true;
    this->initialized = true;
    this->sampled = 0.0;
    this->accepted = 0.0;
    this->samples = _samples;
    this->minibatch = _minibatch;
    this->mask_rate = _mask_rate;
    this->path =_path;
    this->diag = true;
    this->chol = false;
    VectorXd mu = VectorXd::Zero(this->dim);
    if (this->diag){
    	cout << "Diagonal treatment" << endl;
    	VectorXd cov = VectorXd::Ones(this->dim);
    	this->multivariate_gaussian = MVNGaussian(mu, cov);
    }
    else{
    	MatrixXd cov = VectorXd::Ones(this->dim).asDiagonal();
    	this->multivariate_gaussian = MVNGaussian(mu, cov);
    }
    this->multivariate_gaussian.generate(this->chol);
    this->mask_matrix = MatrixXd::Ones(this->samples, this->dim-1);
    this->current_x = VectorXd::Random(this->dim)*0.1;
    if (this->warmup_iterations >= 20){
		this->warmup();
	}
    else{
    	if (this->diag){
			this->inv_diagcov = this->multivariate_gaussian.getdiagInvCov();
			tools.writeToCSVfile(this->path+"warmup_invdiagcov.csv", this->inv_diagcov); //
			tools.writeToCSVfile(this->path+"warmup_diagcov.csv", this->multivariate_gaussian.getdiagCov()); //
		}
		else{
			this->inv_cov = this->multivariate_gaussian.getInvCov();	
			tools.writeToCSVfile(this->path+"warmup_invcov.csv", this->inv_cov); //
			tools.writeToCSVfile(this->path+"warmup_cov.csv", this->multivariate_gaussian.getCov()); //
		}
    }
    this->iterations = _iterations;
    this->mask_mean = 0.0;
    this->mask_std = 0.0;

}

VectorXd Mask_CPU_Hamiltonian_MC::gradient(VectorXd &W, VectorXd &mask,  int n_iter){
	VectorXd grad(W.rows());
	if (this->init_hmc)
	{	
		VectorXd temp = W.tail(W.rows()-1);
		temp = temp.array() * mask.array();
		this->logistic_regression.setWeights(temp);
		this->logistic_regression.setBias(W(0));
		this->logistic_regression.preCompute(n_iter, this->minibatch);
		VectorXd gradWeights = this->logistic_regression.computeGradient();
		double gradBias = this->logistic_regression.getGradientBias();
		grad << gradBias, gradWeights;
		return grad;
	}
	else{
		return grad;
	}
}

double Mask_CPU_Hamiltonian_MC::logPosterior(VectorXd &W,  VectorXd &mask, int n_iter){
	double logPost = 0.0;
	int k=mask.array().sum();
	if (this->init_hmc){
		VectorXd temp = W.tail(W.rows()-1);
		temp = temp.array() * mask.array();
		this->logistic_regression.setWeights(temp);
		this->logistic_regression.setBias(W(0));
		this->logistic_regression.preCompute(n_iter, this->minibatch);
		logPost = -this->logistic_regression.logPosterior() -lgamma(this->dim-1)+lgamma(k)-k*log(this->mask_rate)-(1.0-k)*log(1.0-this->mask_rate);
		return logPost;
	}
	else{
		return logPost;
	}
}


void Mask_CPU_Hamiltonian_MC::run(bool warmup_flag, bool for_predict, double momo){
	if (!warmup_flag and !for_predict) cout << "Run" << endl;
	if (this->init_hmc){	
		VectorXd mask;
		if(!for_predict){
			mask = VectorXd::Ones(this->dim-1);
		}
		else{
			mask = this->mask_matrix.colwise().mean();
		}

		if ((this->iterations > this->samples) and !for_predict){
			this->weights.resize(this->samples, this->dim);	
		}
		else{
			this->weights.resize(this->iterations, this->dim);	
		}

		VectorXd x = this->current_x;
		
		//Hamiltonian
		double Hold;
		double Hnew;
		double Enew;
		double Eold = this->logPosterior(x, mask, 0);

		VectorXd p;

		int n = 0;
		int idx = 0;//
		while (n < this->iterations){
			//if(!for_predict) tools.printProgBar(n, this->iterations);
			p = initial_momentum();

			VectorXd xold = x;
			VectorXd pold = p;

			double epsilon = this->unif(this->step_size);

			if (this->path_length > 0.0) this->num_step = int(this->path_length/epsilon);

			p.noalias() = p - 0.5*epsilon*this->gradient(x, mask, n);
			
			//Leap Frogs
			for (int i = 0; i < this->num_step; ++i){
				x.noalias() = x + epsilon*p;
				if(i == (this->num_step-1)) p.noalias() = p - epsilon*this->gradient(x, mask, n);
			}

			p.noalias() = p - 0.5*epsilon*this->gradient(x, mask, n);

			//Hamiltonian
			Enew = this->logPosterior(x, mask, n);

			if (warmup_flag){
				Hnew = Enew + 0.5 * p.adjoint()*p;
				Hold = Eold + 0.5 * pold.adjoint()*pold;	
			}
			else{
				if (this->diag){
					Hnew = Enew + 0.5 * (p.array()*this->inv_diagcov.array()).matrix().transpose()*p;
					Hold = Eold + 0.5 * (pold.array()*this->inv_diagcov.array()).matrix().transpose()*pold;
				}
				else{
					Hnew = Enew + 0.5 * (p.transpose()*this->inv_cov)*p;
					Hold = Eold + 0.5 * (pold.transpose()*this->inv_cov)*pold;	
				}
			}

			//Metropolis Hasting Correction
			double a = min(0.0, Hold - Hnew);
			if (log(random_uniform()) < a ){
				Eold = Enew;
				this->accepted++;
			}
			else{
				x = xold;	
			}

			if ((this->iterations > this->samples) and !for_predict){
				if (this->iterations-n <= this->samples){ //
					mask_matrix.row(idx) = mask;	
					this->weights.row(idx) = x;
					idx++; //	
				} //
				mask = tools.random_binomial(1, this->mask_rate, this->dim-1);
			}
			else{
				this->weights.row(n) = x;
				if (!for_predict){
					mask = tools.random_binomial(1, this->mask_rate, this->dim-1);	
				}
			}		

			if (n % 1000 == 0) tools.writeToCSVfile(this->path+"dhmc_state.csv", x); //
			
			this->sampled++;

			n++;

		}
		this->current_x = x;
		if (!warmup_flag and !for_predict) tools.writeToCSVfile(this->path+"dhmc_posterior.csv", this->weights); //
		//if(!for_predict)cout << endl;
		if (!warmup_flag and !for_predict){
			this->mask_mean = mask_matrix.rowwise().sum().mean();
			cout << "Mask Mean: "<< mask_mean << endl;	
			this->mask_std = sqrt((mask_matrix.rowwise().sum().array() - this->mask_mean).array().square().sum() / (mask_matrix.rows()));
			cout << "Mask STD: " << mask_std << endl;

		}
		if(!for_predict)this->acceptace_rate();

	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}


VectorXd Mask_CPU_Hamiltonian_MC::predict(MatrixXd &X_test, int psamples, bool simulation, bool data_processing){

	VectorXd predict;
	if (this->init_hmc){
		//int partition = (int)this->iterations*0.5; 
		//MatrixXd temp_weights = this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim);
		MatrixXd temp_weights = this->weights; //
		this->mean_weights = temp_weights.colwise().mean();
		this->sampled = 0.0;
	    this->accepted = 0.0;
	    if(psamples > 1){
			MatrixXd temp_predict(psamples, X_test.rows());
			this->temp_predict_proba.resize(psamples, X_test.rows());
			if (simulation){
				this->multivariate_gaussian = MVNGaussian(temp_weights, this->diag);
				if (this->diag){
					this->inv_diagcov = this->multivariate_gaussian.getInvCov();
					this->multivariate_gaussian.generate();
				}
				else{
					this->inv_cov = this->multivariate_gaussian.getInvCov();	
					this->multivariate_gaussian.generate(this->chol);
				}

				/*this->multivariate_gaussian = MVNGaussian(temp_weights);
				VectorXd mu = VectorXd::Zero(dim);
				this->multivariate_gaussian.setMean(mu);
				this->multivariate_gaussian.generate();*/

				this->iterations = psamples;
				this->run(false, true);
			}
			
			/*VectorXd mean_mask =(this->mask_matrix.colwise().mean());
			VectorXd temp_mask =  this->random_binomial(1, mean_mask, mean_mask.rows());*/

			for (int i = 0; i < psamples; ++i){
				VectorXd W;
				int randNum = rand()%(weights.rows());
				//W = this->weights.row(this->weights.rows()-1-i);	
				W = this->weights.row(randNum);

				VectorXd temp = W.tail(W.rows()-1);
				//temp = temp*0.5;
				this->logistic_regression.setWeights(temp);
				this->logistic_regression.setBias(W(0));
				temp_predict.row(i) = this->logistic_regression.predict(X_test, false, data_processing);
				data_processing = false;
				this->temp_predict_proba.row(i) = this->logistic_regression.predict(X_test, true, data_processing);	
			}
			predict = temp_predict.colwise().mean();
			predict.noalias() = predict.unaryExpr([](double elem){
				return (elem > 0.5) ? 1.0 : 0.0;
			});
		}
		else{
			this->temp_predict_proba.resize(1, X_test.rows());
			VectorXd W = this->mean_weights;
			VectorXd temp = W.tail(W.rows()-1);
			this->logistic_regression.setWeights(temp);
			this->logistic_regression.setBias(W(0));
			predict = this->logistic_regression.predict(X_test, false, data_processing);
			this->temp_predict_proba.row(0) = this->logistic_regression.predict(X_test, true, false);
		}


		return predict;
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}


MatrixXd Mask_CPU_Hamiltonian_MC::get_maskMatrix(){
	if (this->init_hmc){
		return this->mask_matrix;
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
	return MatrixXd::Ones(this->iterations, this->dim-1);
}

void Mask_CPU_Hamiltonian_MC::set_maskMatrix(MatrixXd &_mask_matrix){
	if (this->init_hmc){
		this->mask_matrix = _mask_matrix;
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Mask_CPU_Hamiltonian_MC::getModel(MatrixXd& weights, MatrixXd& mask_matrix, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin){
	weights = this->get_weights();
	mask_matrix = this->get_maskMatrix();
	featureMean = this->logistic_regression.featureMean;
	featureStd = this->logistic_regression.featureStd;
	featureMax = this->logistic_regression.featureMax;
	featureMin = this->logistic_regression.featureMin;
}

void Mask_CPU_Hamiltonian_MC::saveModel(string name){
	tools.writeToCSVfile(this->path+name+"_Model_weights.csv", this->get_weights());
	tools.writeToCSVfile(this->path+name+"_Model_mask.csv", this->get_maskMatrix());
	tools.writeToCSVfile(this->path+name+"_Model_means.csv", this->logistic_regression.featureMean.transpose());
	tools.writeToCSVfile(this->path+name+"_Model_stds.csv", this->logistic_regression.featureStd.transpose());
	tools.writeToCSVfile(this->path+name+"_Model_maxs.csv", this->logistic_regression.featureMax.transpose());
	tools.writeToCSVfile(this->path+name+"_Model_mins.csv", this->logistic_regression.featureMin.transpose());
}

void Mask_CPU_Hamiltonian_MC::loadModel(MatrixXd weights, MatrixXd mask_matrix, VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, bool _normalization, bool  _standarization){
	this->normalization = _normalization;
	this->standarization = _standarization;
	this->logistic_regression.init(this->normalization, this->standarization, true);
	this->set_weightsMatrix(weights);
	this->set_maskMatrix(mask_matrix);
	this->logistic_regression.featureMean = featureMean;
	this->logistic_regression.featureStd = featureStd;
	this->logistic_regression.featureMax = featureMax;
	this->logistic_regression.featureMin = featureMin;
	this->mean_weights = this->weights.colwise().mean();
}

void Mask_CPU_Hamiltonian_MC::setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing){
	this->rows = _X.rows();
	this->logistic_regression.setData(_X, _Y, _preprocesing);
}
void Mask_CPU_Hamiltonian_MC::getMaskStats(double& _mask_mean, double& _mask_std){
	_mask_mean = this->mask_mean;
	_mask_std = this->mask_std;
}
