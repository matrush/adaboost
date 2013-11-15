# all: weak_classifiers process_images

precompute_feature_values: precompute_feature_values.cpp compute_feature.cpp
	@$(CXX) -Wall -o $@ $+

weak_classifiers: weak_classifiers.cpp

process_images: process_images.cpp

%: %.cpp
	@echo "CXX $<"
	@$(CXX) -Wall -O2 -c -o $@ $<

clean:
	@rm -f  *.o
	@rm -rf *.dSYM

.PHONY: all clean
