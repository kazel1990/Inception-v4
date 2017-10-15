CXX = g++
CXXFLAGS = -O2 -std=c++14

all : generator

generator : generator.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean : 
	rm generator train_val.prototxt
