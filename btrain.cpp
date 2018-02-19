#include "brain.hpp"
#include <fstream>



void brain::setTrainConf(const int iterations, const int epochs, const float alpha)
{
	
	this->m_iterations = iterations;
	this->m_alpha = alpha;
	this->m_epochs = epochs;
	
}


void brain::loadTrainConf(void)
{
	
	std::string path("config/trcfg.txt");
	
	std::fstream trcfg(path.c_str(), std::fstream::in);
	if(!trcfg)
	{
		std::cout << "ERROR LOADING TRAINING CONFIG FILE !\n";
		exit(1);
	}
	
	m_alpha = trcfg.get();
	m_epochs = trcfg.get();
	m_iterations = trcfg.get();
	
	
	trcfg.close();
}


void brain::trainOnPop(const std::vector<std::vector<float> > &inputPop, const std::vector<std::vector<float> > &outputPop)
{
	
	int pop((int)outputPop.size());
	
	
	for(int i = 0; i < pop; i++)
	{
		this->brainPropagate(inputPop[i]);
		this->brainRetroPropagate(outputPop[i]);
		this->refitWeights(inputPop[i]);
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
