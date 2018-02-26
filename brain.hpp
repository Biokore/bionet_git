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
	int m_epochs;
	int m_iterations;
	
	
public:
	
	brain();
	brain(int ivs);
	brain(const brain &src);
	~brain() {};
	
	void addLayer(const int pft, const int knb, const float bias);
	void addLayer(const layer src);		// no ref, hard copy it
	
	void brainPropagate(const std::vector<float> &inputsArray);
	void brainBackPropagate(const std::vector<float> &outputsArray);
	void refitWeights(const std::vector<float> &inputsArray);
	
	void setTrainingValues(const float alpha, const int epochs=-1, const int iterations=-1);
	
	
	
	//// GETTERS PART ////
	
	std::string listBrain(void) const;
	std::string listLayer(const int l) const;
	
	std::vector<float> getBrainOutput() const;
	float getBrainOutput(const int k) const;
	float getDelta(const int l, const int k) const;
	int getLayerVectorSize() const;
	int getInputsVectorSize() const;
	layer getLayer(const int i) const;
	
	
	
	//// TRAINING PART ////
	
	void setTrainConf(const int iterations, const int epochs, const float alpha);
	void loadTrainConf(void);
	
	void trainOnPop(const std::vector<std::vector<float> > &inputPop, const std::vector<std::vector<float> > &outputPop);
	float checkError(const std::vector<std::vector<float> > &inputsArray, const std::vector<std::vector<float> > &outputsArray);
	int getMax(const std::vector<float> &testArray);
	
	
};







#endif
