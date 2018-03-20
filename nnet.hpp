#ifndef NNET_HPP
#define NNET_HPP


#include <iostream>
#include <fstream>
#include "layer.hpp"





class nnet
{
	
private:
	
	int m_inputsVectorSize;
	int m_outputsVectorSize;
	
	int m_layersVectorSize;
	int m_fl;		// output layer index
	
	std::vector<layer> m_layersVector;
	
	std::vector<float> m_outputsVector;
	
	float m_error;
	
	float m_alpha;
	float m_momentum;
	
	
public:
	
	nnet();
	nnet(int ivs);
	nnet(const nnet &src);
	~nnet() {};
	
	void setInputVectorSize(const int ivs);
	
	void addLayer(const int pft, const int knb, const float bias);
	void addLayer(const layer src);		// no ref, hard copy it
	
	void propagate(const std::vector<float> &inputsArray);
	void backPropagate(const std::vector<float> &outputsArray);
	void refitWeights(const std::vector<float> &inputsArray);
	
	void save(const std::string name) const;
	void load(const std::string name);
	
	//// GETTERS PART ////
	
	std::string display(void) const;
	std::string display(const int l) const;
	
	std::vector<float> getOutput() const;
	float getOutput(const int k) const;
	float getDelta(const int l, const int k) const;
	int getLayerVectorSize() const;
	int getInputsVectorSize() const;
	layer getLayer(const int i) const;
	
	
	
	//// TRAINING PART ////
	
	void setTrainingValues(const float alpha, const float momentum=.0f);
	void trainOnPop(const std::vector<std::vector<float> > &inputPop, const std::vector<std::vector<float> > &outputPop);
	float checkError(const std::vector<std::vector<float> > &inputsArray, const std::vector<std::vector<float> > &outputsArray);
	int getMax(const std::vector<float> &testArray);
	
	
};







#endif
