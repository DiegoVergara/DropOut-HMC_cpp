//Author: Diego Vergara
#include "CPU_mhmc.hpp"

void CPU_Hamiltonian_MC::init(MatrixXd &_X, VectorXd &_Y, double _lambda, int _warmup_iterations, int _iterations, int _minibatch, double _step_size, int _num_step, bool _normalization, bool _standarization, int _samples, string _path, double _path_length){
	this->lambda=_lambda;
	this->step_size = _step_size;
	this->num_step = _num_step;
	this->path_length = _path_length;
	if (this->path_length > 0.0) this->num_step = int(this->path_length/this->step_size);
	this->warmup_iterations = _warmup_iterations;
	this->X_train = &_X;
 	this->Y_train = &_Y;
	this->normalization = _normalization;
	this->standarization = _standarization;
	this->cols = _X.cols()+1; // bias
	this->rows = _X.rows();
    this->softmax_regression.init(_X, _Y, this->lambda, this->normalization, this->standarization, true);
    this->init_hmc = true;
    this->initialized = true;
    this->sampled = 0.0;
    this->accepted = 0.0;
    this->samples = _samples;
    this->minibatch = _minibatch;
    this->path=_path;
    this->n_classes = this->softmax_regression.getNClasses();
    this->dim = this->n_classes*this->cols;
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
}


VectorXd CPU_Hamiltonian_MC::gradient(VectorXd &W, int n_iter){
	VectorXd grad(W.rows());
	if (this->init_hmc)
	{	
		MatrixXd temp = tools.VtoM(W, this->n_classes, this->cols);
		this->softmax_regression.setWeights(temp);
		this->softmax_regression.preCompute(n_iter, this->minibatch);
		MatrixXd gradWeights = this->softmax_regression.computeGradient();
		grad = tools.MtoV(gradWeights);
		return grad;
	}
	else{
		return grad;
	}
}

double CPU_Hamiltonian_MC::logPosterior(VectorXd &W, int n_iter, bool precompute){
	double logPost = 0.0;
	if (this->init_hmc){
		MatrixXd temp = tools.VtoM(W, this->n_classes, this->cols);
		this->softmax_regression.setWeights(temp);
		if(precompute) this->softmax_regression.preCompute(n_iter, this->minibatch);
		logPost = -this->softmax_regression.logPosterior();
		return logPost;
	}
	else{
		return logPost;
	}
}


void CPU_Hamiltonian_MC::run(bool warmup_flag, bool for_predict, double mom){
	if (!warmup_flag and !for_predict) cout << "Run" << endl;
	if (this->init_hmc){	
		
		VectorXd x = this->current_x;

		if ((this->iterations > this->samples) and !for_predict){
			this->weights.resize(this->samples, this->dim);	
		}
		else{
			this->weights.resize(this->iterations, this->dim);	
		}	
		
		//Hamiltonian
		double Hold;
		double Hnew;
		double Enew;

		double Eold = this->logPosterior(x, 0);

		VectorXd p;

		int n = 0;
		int idx = 0;//
		while (n < this->iterations){
			//if(!for_predict) tools.printProgBar(n, this->iterations);
			p = initial_momentum();//*mom;

			VectorXd xold = x;
			VectorXd pold = p;

			double epsilon = this->unif(this->step_size);

			if (this->path_length > 0.0) this->num_step = int(this->path_length/epsilon);
			p.noalias() = p - 0.5*epsilon*this->gradient(x, n);

			//Leap Frogs
			for (int i = 0; i < this->num_step; ++i){
				x.noalias() = x + epsilon*p;
				if(i == (this->num_step-1)) p.noalias() = p - epsilon*this->gradient(x, n);
			}

			p.noalias() = p - 0.5*epsilon*this->gradient(x, n);
			p = -p;
			//Hamiltonian
			Enew = this->logPosterior(x, n, false);

			if (warmup_flag){
				Hnew = Enew + 0.5 * p.adjoint()*p;
				Hold = Eold + 0.5 * pold.adjoint()*pold;
				//Hnew = Enew + 0.5 * p.transpose()*p;
				//Hold = Eold + 0.5 * pold.transpose()*pold;
			}
			else{
				if (this->diag){
					Hnew = Enew + 0.5 * (p.array()*this->inv_diagcov.array()).matrix().transpose()*p;					
					Hold = Eold + 0.5 * (pold.array()*this->inv_diagcov.array()).matrix().transpose()*pold;
					//Hnew = Enew + 0.5 * this->inv_diagcov.transpose()*p;
					//Hold = Eold + 0.5 * this->inv_diagcov.transpose()*pold;
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
					this->weights.row(idx) = x;
					idx++; //	
				} //
			}
			else{
				this->weights.row(n) = x;
			}	
			

			if (n % 1000 == 0) tools.writeToCSVfile(this->path+"mhmc_state.csv", x); //
			
			this->sampled++;

			n++;

		}
		//cout << endl;
		this->current_x = x;
		if (!warmup_flag and !for_predict) tools.writeToCSVfile(this->path+"hmc_posterior.csv", this->weights); //

		//if(!for_predict)cout << endl;
		if(!for_predict)this->acceptace_rate();
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}


VectorXd CPU_Hamiltonian_MC::predict(MatrixXd &X_test, int psamples, bool simulation, bool data_processing){

	VectorXd predict;
	if (this->init_hmc){
		//int partition = (int)this->iterations*0.5; 
		//MatrixXd temp_weights = this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim);
		MatrixXd temp_weights = this->weights; //
		this->mean_weights = temp_weights.colwise().mean();
		this->sampled = 0.0;
	    this->accepted = 0.0;
	    VectorXd classes = this->softmax_regression.getClasses();
	    tools.writeToCSVfile(this->path+"classes.csv", classes.transpose());
	    bool with_bias = true;
	    if(psamples > 1){
			MatrixXd temp_predict(psamples, X_test.rows()*this->n_classes);
			this->predict_history.resize(X_test.rows(), psamples);
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
			bool with_bias = true;
			for (int i = 0; i < psamples; ++i){
				VectorXd W;
				int randNum = rand()%(weights.rows());
				//W = this->weights.row(this->weights.rows()-1-i);	
				W = this->weights.row(randNum);	

				MatrixXd temp = tools.VtoM(W, this->n_classes, this->cols);
				this->softmax_regression.setWeights(temp);
				MatrixXd temp2  = this->softmax_regression.predict_proba(X_test, data_processing, with_bias);
				stringstream ss; //
				ss << i; //
				tools.writeToCSVfile(this->path+ss.str()+"_essemble.csv", temp2); //
				temp_predict.row(i)  = tools.MtoV(temp2);
				data_processing = false;
				with_bias = false;
				this->predict_history.col(i) = this->softmax_regression.predict(X_test, data_processing, with_bias);
			}
			tools.writeToCSVfile(this->path+"temp_predict.csv", temp_predict); //
			VectorXd predict_essemble_mean = temp_predict.colwise().mean();
			VectorXd predict_essemble_std = ((temp_predict.rowwise() - predict_essemble_mean.transpose()).array().square().colwise().sum() / (temp_predict.rows())).sqrt();
			this->temp_predict_proba = tools.VtoM(predict_essemble_mean, X_test.rows(), this->n_classes);
			this->temp_predict_proba_std = tools.VtoM(predict_essemble_std, X_test.rows(), this->n_classes);
			predict = tools.argMax(this->temp_predict_proba, true);
			for (int i = 0; i < predict.size(); ++i) predict(i) = classes(predict(i));
		}
		else{
			VectorXd W = this->mean_weights;
			MatrixXd temp = tools.VtoM(W, this->n_classes, this->cols);
			this->softmax_regression.setWeights(temp);
			predict = this->softmax_regression.predict(X_test, data_processing, true);
		}
		
		return predict;
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}


void CPU_Hamiltonian_MC::getModel(MatrixXd& weights, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin){
	weights = this->get_weights();
	featureMean = this->softmax_regression.featureMean;
	featureStd = this->softmax_regression.featureStd;
	featureMax = this->softmax_regression.featureMax;
	featureMin = this->softmax_regression.featureMin;
}

void CPU_Hamiltonian_MC::saveModel(string name){
	tools.writeToCSVfile(this->path+name+"_Model_weights.csv", this->get_weights());
	tools.writeToCSVfile(this->path+name+"_Model_means.csv", this->softmax_regression.featureMean.transpose());
	tools.writeToCSVfile(this->path+name+"_Model_stds.csv", this->softmax_regression.featureStd.transpose());
	tools.writeToCSVfile(this->path+name+"_Model_maxs.csv", this->softmax_regression.featureMax.transpose());
	tools.writeToCSVfile(this->path+name+"_Model_mins.csv", this->softmax_regression.featureMin.transpose());
}

void CPU_Hamiltonian_MC::loadModel(MatrixXd weights, VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin,  bool _normalization, bool _standarization){
	this->normalization = _normalization;
	this->standarization = _standarization;
	this->softmax_regression.init(this->normalization, this->standarization, true);
	this->set_weightsMatrix(weights);
	this->softmax_regression.featureMean = featureMean;
	this->softmax_regression.featureStd = featureStd;
	this->softmax_regression.featureMax = featureMax;
	this->softmax_regression.featureMin = featureMin;
	this->mean_weights = this->weights.colwise().mean();
}

void CPU_Hamiltonian_MC::setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing){
	this->softmax_regression.setData(_X, _Y, _preprocesing);
}

