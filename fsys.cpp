#include "fsys.hpp"




IDX_class::IDX_class()
{
	m_type = 0;
	m_dim = 0;
	m_popNumber = 0;
	
}


IDX_class::IDX_class(const std::string &pics, const std::string &labels)
{
	m_type = 0;
	m_dim = 0;
	m_popNumber = 0;
	
	openFiles(pics, labels);
	
}


IDX_class::~IDX_class()
{
	
	
}





bool IDX_class::openFiles(const std::string &pics, const std::string &labels)
{
	std::fstream picsFile(pics.c_str(), std::fstream::in);
	std::fstream labelsFile(labels.c_str(), std::fstream::in);
	
	if(!picsFile || !labelsFile)
	{
		std::cout << "ERR\n";
		return false;
	}
	
	picsFile.get();		// zero
	picsFile.get();		// zero
	
	labelsFile.get();	// zero
	labelsFile.get();	// zero
	labelsFile.get();	// type
	labelsFile.get();	// dim
	labelsFile.get();	// dim_hhb
	labelsFile.get();	// dim_hlb
	labelsFile.get();	// dim_lhb
	labelsFile.get();	// dim_llb
	
	m_type = (int)picsFile.get();
	m_dim = (int)picsFile.get();
	
	m_inputDim.resize(m_dim);
	
	// 1ere dimension = taille de la popula tion
	// seconde et troisième dimension = taille de l'image
	for(int d = 0; d < m_dim; d++)
	{
		m_inputDim[d] = 256*256*256*(int)picsFile.get() + 256*256*(int)picsFile.get() + 256*(int)picsFile.get() + (int)picsFile.get();
	}
	
	m_popNumber = m_inputDim[0];
	m_inputVectorSize = m_inputDim[1] * m_inputDim[2];
	
	m_inputs.resize(m_popNumber);
	m_outputs.resize(m_popNumber);
	
	for(int i = 0; i < m_popNumber; i++)
	{
		for(int v = 0; v < m_inputVectorSize; v++)
		{
			m_inputs[i].push_back((float)((int)picsFile.get())/255.0);
		}
		m_outputs[i].resize(10, 0.0);
		m_outputs[i][(int)labelsFile.get()] = 1.0;
	}
	
	
	picsFile.close();
	labelsFile.close();
	
	return true;
}



void IDX_class::printPic(int index)
{
	int wi(this->m_inputDim[1]);
	int he(this->m_inputDim[2]);
	
	
	for(int h = 0; h < he; h++)
	{
		for(int w = 0; w < wi; w++)
		{
			if(m_inputs[index][w + (h*wi)] > 0.0)
			{
				std::cout << " x";
			}
			else
			{
				std::cout << " .";
			}
		}
		std::cout << std::endl;
	}
	
}


std::vector<float> IDX_class::getInputsArray(int index) const
{
	
	return m_inputs[index];
}


std::vector<float> IDX_class::getOutputsArray(int index) const
{
	
	return m_outputs[index];
}


std::vector<std::vector<float> > IDX_class::getInputsArray(void) const
{
	
	return m_inputs;
}


std::vector<std::vector<float> > IDX_class::getOutputsArray(void) const
{
	
	return m_outputs;
}










// END
