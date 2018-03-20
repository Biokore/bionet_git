#include "layer.hpp"








int layer::getPropFunction() const
{
	
	return this->m_propType;
}


int layer::getInputsVectorSize() const
{
	
	return this->m_inputsVectorSize;
}


int layer::getKernelsVectorSize() const
{
	
	return this->m_kernelsVectorSize;
}





float layer::getBias()const
{
	
	return this->m_bias;
}


float layer::getWeight(const int k, const int i) const
{
	
	return this->m_weights[k][i];
}


float layer::getAgregValue(const int k) const
{
	
	return this->m_agregValue[k];
}


float layer::getOutput(const int k) const
{
	
	return this->m_propValue[k];
}


float layer::getDelta(const int k) const
{
	
	return this->m_delta[k];
}





std::vector<float> layer::getOutput() const
{
	
	return this->m_propValue;
}


std::vector<float> layer::getDelta() const
{
	
	return this->m_delta;
}





std::string layer::display() const
{
	std::ostringstream oss;
	
	oss << "LAYER TYPE = " << this->m_propType << '\n';
	oss << "INPUTS NUMBER = " << this->m_inputsVectorSize * m_kernelsVectorSize << '\n';
	oss << "KERNELS NUMBER = " << this->m_kernelsVectorSize << '\n';
	oss << "BIAS VALUE = " << this->m_bias << '\n';
	
	return oss.str();
}
	















// END
