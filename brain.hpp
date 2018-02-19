#ifndef BRAIN_HPP
#define BRAIN_HPP


#include "layer.hpp"





class brain
{
	
private:
	
	int m_inputsVectorSize;
	int m_outputsVectorSize;
	
	int m_layersVectorSize;
	int m_fl;		// output layer index
	
	std::vector<layer> m_layersVector;
	
	std::vector<float> m_inputsVector;
	std::vector<float> m_outputsVector;
	
	float m_error;
	
	float m_alpha;
	int m_iterations;
	
	
public:
	
	brain(void);
	brain(int ivs);
	~brain(void);
	
	void addLayer(const int pft, const int knb, const float bias);
	
	void brainPropagate(const std::vector<float> &inputsArray);
	void brainRetroPropagate(const std::vector<float> &outputsArray);
	void refitWeights(const std::vector<float> &inputsArray);
	
	float checkError(const std::vector<std::vector<float> > &inputsArray, const std::vector<std::vector<float> > &outputsArray);
	int getMax(const std::vector<float> &testArray);
	
	
	
	std::string listBrain(void) const;
	std::string listLayer(const int l) const;
	
	std::vector<float> getBrainOutput() const;
	float getBrainOutput(const int k) const;
	float getDelta(const int l, const int k) const;
	
};







#endif
