#ifndef FSYS_H
#define FSYS_H

#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>


class IDX_class
{
	
private:
	
	int m_type;
	int m_dim;
	int m_popNumber;
	int m_inputVectorSize;
	int m_outputSize;
	
	std::vector<int> m_inputDim;
	
	std::vector<std::vector<float> > m_inputs;
	std::vector<std::vector<float> > m_outputs;
	
	
public:
	
	IDX_class();
	IDX_class(const std::string &pics, const std::string &labels, const int outputSize);
	~IDX_class();
	
	bool openFiles(const std::string &pics, const std::string &labels);
	void printPic(int index);
	
	
	std::vector<float> getInputsArray(int index) const;
	std::vector<float> getOutputsArray(int index) const;
	std::vector<std::vector<float> > getInputsArray(void) const;
	std::vector<std::vector<float> > getOutputsArray(void) const;
	int getPopNumber(void) const;
};





#endif
