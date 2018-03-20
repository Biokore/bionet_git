#include "nnet.hpp"
#include <fstream>



void nnet::setTrainingValues(const float alpha, const float momentum)
{
	
	this->m_alpha = alpha;
	this->m_momentum = momentum;
	
}


void nnet::trainOnPop(const std::vector<std::vector<float> > &inputPop, const std::vector<std::vector<float> > &outputPop)
{
	
	int pop((int)outputPop.size());
	
	
	for(int i = 0; i < pop; i++)
	{
		this->propagate(inputPop[i]);
		this->backPropagate(outputPop[i]);
		this->refitWeights(inputPop[i]);
	}
}


float nnet::checkError(const std::vector<std::vector<float> > &inputsArray, const std::vector<std::vector<float> > &outputsArray)
{
	m_error = 0.0;
	int pop((int)outputsArray.size());
	
	for(int o = 0; o < pop; o++)
	{
		this->propagate(inputsArray[o]);
		
		if(getMax(this->getOutput()) != getMax(outputsArray[o]))
		{
			m_error += 1.0/pop;
		}
	}
	
	return m_error;
}


int nnet::getMax(const std::vector<float> &testArray)
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
