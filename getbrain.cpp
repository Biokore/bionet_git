#include "brain.hpp"








std::string brain::listBrain(void) const
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


std::string brain::listLayer(const int l) const
{
	
	return (this->m_layersVector[l].listLayer());
}





std::vector<float> brain::getBrainOutput() const
{
	
	return (this->m_layersVector[m_fl].getLayerOutput());
}


float brain::getBrainOutput(const int k) const
{
	
	return (this->m_layersVector[m_fl].getLayerOutput(k));
}


float brain::getDelta(const int l, const int k) const
{
	
	return (this->m_layersVector[l].getDelta(k));
}


int brain::getLayerVectorSize() const
{
	
	return this->m_layersVectorSize;
}


int brain::getInputsVectorSize() const
{
	
	
	return this->m_inputsVectorSize;
}


layer brain::getLayer(const int i) const
{
	
	return this->m_layersVector[i];
}













// END
