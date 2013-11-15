# all: weak_classifiers read_images

precompute_feature_values: precompute_feature_values.cpp compute_feature.cpp
	@$(CXX) -Wall -O2 -o $@ $+ utils.cpp

weak_classifiers: weak_classifiers.cpp

process_images: process_images.cpp

%: %.cpp
	@$(CXX) -Wall -O2 -o $@ $< utils.cpp

clean:
	@rm -f  *.o
	@rm -rf *.dSYM

.PHONY: all clean
