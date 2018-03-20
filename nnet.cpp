#include "nnet.hpp"





nnet::nnet():
		m_inputsVectorSize(0), m_outputsVectorSize(0), m_layersVectorSize(0), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_momentum(.0f)
{
	
	
}


nnet::nnet(const int ivs):
		m_inputsVectorSize(ivs), m_outputsVectorSize(0), m_layersVectorSize(0), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_momentum(.0f)
{
	
	
}


nnet::nnet(const nnet &src):
		m_inputsVectorSize(src.getInputsVectorSize()), m_outputsVectorSize(0), m_layersVectorSize(src.getLayerVectorSize()), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_momentum(.0f)
{
	
	for(int i = 0; i < this->m_layersVectorSize; i++)
	{
		this->addLayer(src.getLayer(i));
	}
}


// nnet::~nnet() {}





void nnet::setInputVectorSize(const int ivs)
{
	
	this->m_inputsVectorSize = ivs;
}


void nnet::addLayer(const int pft, const int knb, const float bias)
{
	int ivs(0);
	
	if(m_layersVectorSize == 0)
	{
		ivs = m_inputsVectorSize;
		
	}
	else
	{
		ivs = (m_layersVector[m_layersVectorSize-1].getKernelsVectorSize());
	}
		
	layer l(pft, ivs, knb, bias);
	m_layersVector.push_back(l);
	
	m_outputsVectorSize = knb;
	m_fl = m_layersVectorSize;
	m_layersVectorSize++;
}


void nnet::addLayer(const layer src)		// no ref, hard copy it
{
	
	this->m_layersVector.push_back(src);
	this->m_outputsVectorSize = src.getKernelsVectorSize();
}


void nnet::propagate(const std::vector<float> &inputsArray)
{
	this->m_layersVector[0].propagate(inputsArray);
	
	for(int l = 1; l < m_layersVectorSize; l++)
	{
		this->m_layersVector[l].propagate((this->m_layersVector[l-1].getOutput()));
	}
}


void nnet::backPropagate(const std::vector<float> &outputsArray)
{
	std::vector<float> error;
	float t_error(0.0f);
	
	for(int o = 0; o < m_outputsVectorSize; o++)
	{
		error.push_back(outputsArray[o] - m_layersVector[m_fl].getOutput(o));
	}
	
	m_layersVector[m_fl].calculateDelta(error);
	error.clear();
	
	
	int ivs(0);
	int knb(0);
	
	for(int l = m_fl; l > 0; l--)
	{
		ivs = m_layersVector[l].getInputsVectorSize();
		knb = m_layersVector[l].getKernelsVectorSize();
		
		for(int i = 0; i < ivs; i++)
		{
			for(int k = 0; k < knb; k++)
			{
				t_error += m_layersVector[l].getWeight(k, i+1) * m_layersVector[l].getDelta(k);
			}
			error.push_back(t_error);
			t_error = 0.0f;
		}
		m_layersVector[l-1].calculateDelta(error);
		error.clear();
	}
}


void nnet::refitWeights(const std::vector<float> &inputsArray)
{
	m_layersVector[0].refitWeights(inputsArray, m_alpha);
	
	for(int l = 1; l < m_layersVectorSize; l++)
	{
		m_layersVector[l].refitWeights(m_layersVector[l-1].getOutput(), m_alpha);
	}
	
	
}


void nnet::save(const std::string name) const
{
	std::fstream nnFile(name.c_str(), std::fstream::out);
	if(!nnFile)
	{
		std::cerr << "ERROR WHILE SAVING FILE " << name << std::endl;
		exit(1);
	}
	
	nnFile << m_inputsVectorSize;
	nnFile << m_layersVectorSize;
	nnFile << m_alpha;
	nnFile << m_momentum;
	
	for(int l = 0; l < m_layersVectorSize; l++)
	{
		
		nnFile << m_layersVector[l].getPropFunction();
		nnFile << m_layersVector[l].getKernelsVectorSize();
		nnFile << m_layersVector[l].getBias();
		
		for(int k = 0; k < m_layersVector[l].getKernelsVectorSize(); k++)
		{
			for(int w = 0; w < m_layersVector[l].getInputsVectorSize(); w++)
			{
				nnFile << m_layersVector[l].getWeight(k, w);
			}
		}
	}
	
	
	std::cout << "File " << name << "saved..." << std::endl;
	
	
	nnFile.close();
}


void nnet::load(const std::string name)
{
	std::fstream nnFile(name.c_str(), std::fstream::in);
	
	if(!nnFile)
	{
		std::cerr << "ERROR WHILE LOADING FILE " << name << std::endl;
		exit(1);
	}
	
	int pft(0);
	int knb(0);
	float bias(.0f);
	
	float val(.0f);
	
	nnFile >> m_inputsVectorSize;
	std::cout << "ivs : " << m_inputsVectorSize << std::endl;
	nnFile >> m_layersVectorSize;
	std::cout << "lnb : " << m_layersVectorSize << std::endl;
	nnFile >> m_alpha;
	std::cout << "alpha : " << m_alpha << std::endl;
	nnFile >> m_momentum;
	std::cout << "momentum: " << m_momentum << std::endl;
	
	for(int l = 0; l < m_layersVectorSize; l++)
	{
		nnFile >> pft;
		nnFile >> knb;
		nnFile >> bias;
		
		this->addLayer(pft, knb, bias);
		
		for(int k = 0; k < knb; k++)
		{
			for(int w = 0; w < m_layersVector[l].getInputsVectorSize(); w++)
			{
				nnFile >> val;
				m_layersVector[l].setWeights(k, w, val);
			}
		}
		
		std::cout << "layer " << l << " ok\n";
	}
	
	
	
	nnFile.close();
	
}
























// END
