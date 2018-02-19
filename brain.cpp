#include "brain.hpp"





brain::brain(void):
		m_inputsVectorSize(0), m_outputsVectorSize(0), m_layersVectorSize(0), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_iterations(25)
{
	
	
}


brain::brain(const int ivs):
		m_inputsVectorSize(ivs), m_outputsVectorSize(0), m_layersVectorSize(0), m_fl(0), 
		m_error(0.0f), m_alpha(0.05f), m_iterations(25)
{
	
	m_inputsVector.resize(m_inputsVectorSize);
}


brain::~brain(void)
{
	
}





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


void brain::brainPropagate(const std::vector<float> &inputsArray)
{
	this->m_layersVector[0].layerPropagate(inputsArray);
	
	for(int l = 1; l < m_layersVectorSize; l++)
	{
		this->m_layersVector[l].layerPropagate((this->m_layersVector[l-1].getLayerOutput()));
	}
}


void brain::brainRetroPropagate(const std::vector<float> &outputsArray)
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


float brain::checkError(const std::vector<std::vector<float> > &inputsArray, const std::vector<std::vector<float> > &outputsArray)
{
	m_error = 0.0;
	int pop((int)outputsArray.size());
	
	for(int o = 0; o < pop; o++)
	{
		this->brainPropagate(inputsArray[o]);
		
		if(getMax(this->getBrainOutput()) != getMax(outputsArray[o]))
		{
			m_error += 1.0/pop;
		}
	}
	
	return m_error;
}


int brain::getMax(const std::vector<float> &testArray)
{
	int asize((int)testArray.size());
	
	int ret(0);
	float max(0.0f);
	
	for(int i = 0; i < asize; i++)
	{
		if(testArray[i] > max)
		{
			max = testArray[i];
			ret = i;
		}
	}
	
	
	return ret;
}





















// END
