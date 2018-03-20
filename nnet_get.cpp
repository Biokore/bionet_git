#include "nnet.hpp"








std::string nnet::display(void) const
{
	std::ostringstream oss;
	
	for(int l = 0; l < m_layersVectorSize; l++)
	{
		oss << "LAYER N°" << l << " : ";
		oss << "TYPE = " << this->m_layersVector[l].getPropFunction() << " ";
		oss << "KERNELS = " << this->m_layersVector[l].getKernelsVectorSize() << " ";
		oss << "INPUTS = " << this->m_layersVector[l].getInputsVectorSize() * this->m_layersVector[l].getKernelsVectorSize() << "\n";
	}
	
	return oss.str();
}


std::string nnet::display(const int l) const
{
	
	return (this->m_layersVector[l].display());
}





std::vector<float> nnet::getOutput() const
{
	
	return (this->m_layersVector[m_fl].getOutput());
}


float nnet::getOutput(const int k) const
{
	
	return (this->m_layersVector[m_fl].getOutput(k));
}


float nnet::getDelta(const int l, const int k) const
{
	
	return (this->m_layersVector[l].getDelta(k));
}


int nnet::getLayerVectorSize() const
{
	
	return this->m_layersVectorSize;
}


int nnet::getInputsVectorSize() const
{
	
	
	return this->m_inputsVectorSize;
}


layer nnet::getLayer(const int i) const
{
	
	return this->m_layersVector[i];
}













// END
