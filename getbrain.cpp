#include "brain.hpp"








std::string brain::listBrain(void) const
{
	std::ostringstream oss;
	
	for(int l = 0; l < m_layersVectorSize; l++)
	{
		oss << "LAYER N°" << l << " : ";
		oss << "TYPE = " << this->m_layersVector[l].getPropFunction() << " ";
		oss << "KERNELS = " << this->m_layersVector[l].getKernelsVectorSize() << " ";
		oss << "INPUTS = " << this->m_layersVector[l].getInputsVectorSize() << "\n";
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













// END
