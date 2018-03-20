#ifndef LAYER_HPP
#define LAYER_HPP


#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <cmath>

#define THRESHOLD	0
#define LINEAR		1
#define RELU		2
#define RELUS		3
#define SIGMOID		4
#define HTAN		5

#define INPUT		true
#define STD			false



class layer
{
	
private:
	
	int m_propType;
	
	int m_inputsVectorSize;
	int m_kernelsVectorSize;
	
	float m_bias;
	
	
	std::vector<std::vector<float> > m_weights;
	std::vector<float> m_agregValue;
	std::vector<float> m_propValue;
	
	std::vector<float> m_delta;
	
	
	
public:
	
	layer();
	layer(const int pft, const int ivs, const int knb, const float bias);
	~layer();
	
	void initWeights();
	void setWeights(const int k, const int w, const float val);
	
	void propagate(const std::vector<float> &inputsArray);
	
	void calculateDelta(const layer &lay);	// ATTENTION : fonction à vérifier / tester
	void calculateDelta(const std::vector<float> &error);
	void refitWeights(const std::vector<float> &inputsArray, const float alpha);
	
// 	float (layer::*calculatePropValue[])(const float av);
// 	float (layer::*calculateDerivate[])(const float av);
	
	
	
	int getPropFunction() const;
	int getInputsVectorSize() const;
	int getKernelsVectorSize() const;
	
	float getBias()const;
	float getWeight(const int k, const int i) const;
	float getAgregValue(const int k) const;
	float getOutput(const int k) const;
	float getDelta(const int k) const;
	
	std::vector<float> getOutput() const;
	std::vector<float> getDelta() const;
	
	std::string display() const;
	
	
	
	
	
	float threshold(const float av);
	float linear(const float av);
	float relu(const float av);
	float relus(const float av);
	float sigmoid(const float av);
	float htan(const float av);
	
	float threshold_d(const float av);
	float linear_d(const float av);
	float relu_d(const float av);
	float relus_d(const float av);
	float sigmoid_d(const float av);
	float htan_d(const float av);
	
	
};









#endif
