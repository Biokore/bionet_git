#include "layer.hpp"


float (layer::*calculatePropValue[6])(const float av) = {&layer::threshold, &layer::linear, &layer::relu, &layer::relus, &layer::sigmoid, &layer::htan};

float (layer::*calculateDerivate[6])(const float av) = {&layer::threshold_d, &layer::linear_d, &layer::relu_d, &layer::relus_d, &layer::sigmoid_d, &layer::htan_d};


layer::layer()
{
	
	
}


layer::layer(const int pft, const int ivs, const int knb, const float bias):
		m_propType(pft), m_inputsVectorSize(ivs), m_kernelsVectorSize(knb), m_bias(bias)
{
	
	m_agregValue.resize(m_kernelsVectorSize);
	m_propValue.resize(m_kernelsVectorSize);
	
	m_delta.resize(m_kernelsVectorSize);
	
	m_weights.resize(m_kernelsVectorSize);
	
	for(int k = 0; k < m_kernelsVectorSize; k++)
	{
		m_weights[k].resize(m_inputsVectorSize+1);
	}
	
	this->initWeights();
	
}


layer::~layer()
{
	
	
}




void layer::initWeights()
{
	float w = sqrt(3.0/(float)m_inputsVectorSize);
	float imax = w;
	float imin = -w;
	
	for(int k = 0; k < m_kernelsVectorSize; k++)
	{
		for(int i = 0; i < m_inputsVectorSize+1; i++)
		{
			m_weights[k][i] = ((float)rand() / (RAND_MAX/(imax - imin))) + imin;
		}
	}
}





void layer::layerPropagate(const std::vector<float> &inputsArray)
{
	for(int k = 0; k < m_kernelsVectorSize; k++)
	{
		m_agregValue[k] = m_weights[k][0] * m_bias;
		
		for(int i = 0; i < m_inputsVectorSize; i++)
		{
			m_agregValue[k] += m_weights[k][i+1] * inputsArray[i];
		}
		
		m_propValue[k] = (this->*calculatePropValue[m_propType])(m_agregValue[k]);
	}
}

//// ATTENTION : pas de coef alpha et pas de passage sur de la couche
void layer::calculateDelta(const layer &lay)
{
	float error(0.0);
	std::cout << lay.getLayerOutput(1) << std::endl;
	for(int k = 0; k < m_kernelsVectorSize; k++)
	{
		m_delta[k] = (this->*calculateDerivate[m_propType])(m_agregValue[k]);
		
		for(int j = 0; j < lay.getKernelsVectorSize(); j++)
		{
			error = lay.getDelta(j) * lay.getWeight(j, k+1);
			m_delta[k] *= error;
		}
	}
}


void layer::calculateDelta(const std::vector<float> &error)
{
	
	for(int k = 0; k < m_kernelsVectorSize; k++)
	{
		m_delta[k] = (this->*calculateDerivate[m_propType])(m_agregValue[k]) * error[k];
	}
}


void layer::refitWeights(const std::vector<float> &inputsArray, const float alpha)
{
	for(int k = 0; k < m_kernelsVectorSize; k++)
	{
		m_weights[k][0] = m_weights[k][0] + alpha * m_delta[k] * m_bias;
		
		for(int i = 0; i < m_inputsVectorSize; i++)
		{
			m_weights[k][i+1] = m_weights[k][i+1] + alpha * m_delta[k] * inputsArray[i];
		}
	}
}












float layer::threshold(const float av)
{
	if(av <= 0.0)
	{
		return 0.0;
	}
	else
	{
		return 1.0;
	}
}


float layer::linear(const float av)
{
	return av;
}


float layer::relu(const float av)
{
	if(av <= 0.0)
	{
		return 0.0;
	}
	else
	{
		return av;
	}
}


float layer::relus(const float av)
{
	if(av <= 0.0)
	{
		return 0.0;
	}
	else if(av >= 1.0)
	{
		return 1.0;
	}
	else
	{
		return av;
	}
}


float layer::sigmoid(const float av)
{
	return (1/(1+exp(-av)));
}


float layer::htan(float av)
{
	return ((2/(1+exp(-2*av)))-1);
}


float layer::threshold_d (const float av)		// voir quoi retourner exactement si x <= 0
{
	if (av>0.0)
	{
		return 0.0;
	}
	else
	{
		return 0.5;
	}
}


float layer::linear_d (const float av)
{
	float a(av);
	a = a;
	return 1;
}


float layer::relu_d (const float av)
{
	if (av>0.0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


float layer::relus_d (const float av)
{
	if (av>0.0 && av < 1.0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


float layer::sigmoid_d (const float av)
{
	float xd = sigmoid(av);
	return (xd*(1-xd));
}


float layer::htan_d (const float av)
{
	float xd = htan(av);
	return (1-(xd*xd));
}














// END
