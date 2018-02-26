#include "brain.hpp"





brain::brain():
		m_inputsVectorSize(0), m_outputsVectorSize(0), m_layersVectorSize(0), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_epochs(1), m_iterations(25)
{
	
	
}


brain::brain(const int ivs):
		m_inputsVectorSize(ivs), m_outputsVectorSize(0), m_layersVectorSize(0), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_epochs(1), m_iterations(25)
{
	
	m_inputsVector.resize(m_inputsVectorSize);		// useless ; can be removed...
}


brain::brain(const brain &src):
		m_inputsVectorSize(src.getInputsVectorSize()), m_outputsVectorSize(0), m_layersVectorSize(src.getLayerVectorSize()), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_epochs(1), m_iterations(25)
{
	
	m_inputsVector.resize(m_inputsVectorSize);		// useless ; can be removed...
	
	for(int i = 0; i < this->m_inputsVectorSize; i++)
	{
		this->addLayer(src.getLayer(i));
	}
}


// brain::~brain() {}





void brain::addLayer(const int pft, const int knb, const float bias)
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


void brain::addLayer(const layer src)		// no ref, hard copy it
{
	
	this->m_layersVector.push_back(src);
	this->m_outputsVectorSize = src.getKernelsVectorSize();
}


void brain::brainPropagate(const std::vector<float> &inputsArray)
{
	this->m_layersVector[0].layerPropagate(inputsArray);
	
	for(int l = 1; l < m_layersVectorSize; l++)
	{
		this->m_layersVector[l].layerPropagate((this->m_layersVector[l-1].getLayerOutput()));
	}
}


void brain::brainBackPropagate(const std::vector<float> &outputsArray)
{
	std::vector<float> error;
	float t_error(0.0f);
	
	for(int o = 0; o < m_outputsVectorSize; o++)
	{
		error.push_back(outputsArray[o] - m_layersVector[m_fl].getLayerOutput(o));
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


void brain::refitWeights(const std::vector<float> &inputsArray)
{
	m_layersVector[0].refitWeights(inputsArray, m_alpha);
	
	for(int l = 1; l < m_layersVectorSize; l++)
	{
		m_layersVector[l].refitWeights(m_layersVector[l-1].getLayerOutput(), m_alpha);
	}
}


void brain::setTrainingValues(const float alpha, const int epochs, const int iterations)
{
	
	this->m_alpha = alpha;
	
	if(epochs >= 0)
	{
		this->m_epochs = epochs;
	}
	
	if(iterations >=0)
	{
		this->m_iterations = iterations;
	}
	
}





















// END
