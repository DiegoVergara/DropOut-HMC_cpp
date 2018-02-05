//Author: Diego Vergara
#include "hmc.hpp"


Hamiltonian_MC::Hamiltonian_MC(){
	this->init_hmc = true;
}

void Hamiltonian_MC::warmup(){
	if (this->init_hmc){
		cout << "WarMup" << endl;
		this->iterations = this->warmup_iterations;
		this->run(true);
		this->sampled = 0.0;
	    this->accepted = 0.0;
	    MatrixXd temp_weights = this->weights.block(int(this->weights.rows()/10),0,this->weights.rows()- int(this->weights.rows()/10),this->weights.cols());
		this->multivariate_gaussian = MVNGaussian(temp_weights, this->diag);
		VectorXd mu = VectorXd::Zero(dim);
		this->multivariate_gaussian.setMean(mu);
		this->chol = false;
		if (this->diag){
			this->inv_diagcov = this->multivariate_gaussian.getdiagInvCov();
			tools.writeToCSVfile(this->path+"warmup_invdiagcov.csv", this->inv_diagcov); //
			tools.writeToCSVfile(this->path+"warmup_diagcov.csv", this->multivariate_gaussian.getdiagCov()); //
			this->multivariate_gaussian.generate();
		}
		else{
			this->inv_cov = this->multivariate_gaussian.getInvCov();	
			tools.writeToCSVfile(this->path+"warmup_invcov.csv", this->inv_cov); //
			tools.writeToCSVfile(this->path+"warmup_cov.csv", this->multivariate_gaussian.getCov()); //
			this->multivariate_gaussian.generate(this->chol);
		}
		this->current_x = VectorXd::Random(this->dim)*0.1;
		tools.writeToCSVfile(this->path+"warmup_mean.csv", mu); //
		
	}
}

void Hamiltonian_MC::acceptace_rate(){
	cout << "Acceptace Rate: "<< 100 * (float) this->accepted/this->sampled <<" %" <<endl;
}


VectorXd Hamiltonian_MC::initial_momentum(){
	return this->multivariate_gaussian.sample();
}

double Hamiltonian_MC::unif(double step_size){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.9, 1.1);
    return step_size * dis(gen);
}

VectorXd Hamiltonian_MC::random_generator(int dimension){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  	mt19937 generator;
  	generator.seed(seed1);
  	normal_distribution<double> dnormal(0.0,1.0);
	VectorXd random_vector(dimension);

	for (int i = 0; i < dimension; ++i){
		random_vector(i) = dnormal(generator);
	}
	return random_vector;
}

double Hamiltonian_MC::random_uniform(){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

VectorXd Hamiltonian_MC::random_binomial(int n, VectorXd prob, int dim){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  	mt19937 generator;
  	generator.seed(seed1);
	VectorXd random_vector(dim);
	for (int i = 0; i < dim; ++i){
		binomial_distribution<int> dbinomial(n,prob(i));
		random_vector(i) = dbinomial(generator);
	}
	return random_vector;
}


double Hamiltonian_MC::avsigmaGauss(double mean, double var){
  double erflambda = sqrt(M_PI)/4;
  double out = 0.5+0.5*erf(erflambda*mean/sqrt(1+2*pow(erflambda,2)*var));
  return out;
}

VectorXd Hamiltonian_MC::cumGauss(VectorXd &w, MatrixXd &phi, MatrixXd &Smat){
  	int M = phi.rows();
  	VectorXd ptrain(M);
  	//VectorXd weights = w.tail(w.rows()-1);
  	//double bias = w(0);
  	//#pragma omp parallel for schedule(static)
	for (int i = 0; i < M; ++i){
	  double mean = w.dot(phi.row(i));
	  double var = (phi.row(i) * Smat * phi.row(i).transpose())(0);
	  ptrain(i) = avsigmaGauss(mean, var);
	}
    return ptrain;
}

MatrixXd Hamiltonian_MC::get_weights(){
	MatrixXd weights;

	if (this->init_hmc){
		return this->weights;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return weights;
	}
}

MatrixXd Hamiltonian_MC::get_predict_proba(){
	MatrixXd predict_proba;

	if (this->init_hmc){
		return this->temp_predict_proba;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict_proba;
	}
}

MatrixXd Hamiltonian_MC::get_predict_proba_std(){
	MatrixXd predict_proba;

	if (this->init_hmc){
		return this->temp_predict_proba_std;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict_proba;
	}
}

MatrixXd Hamiltonian_MC::get_predict_history(){
	MatrixXd predict_proba;

	if (this->init_hmc){
		return this->predict_history;
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict_proba;
	}
}

void Hamiltonian_MC::set_weights(VectorXd &_weights){
	if (this->init_hmc){	
		this->mean_weights = _weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Hamiltonian_MC::set_weightsMatrix(MatrixXd &_weights){
	if (this->init_hmc){	
		this->weights = _weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Hamiltonian_MC::set_iterations(int _iterations){
	this->iterations = _iterations;
}


