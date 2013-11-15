# all: weak_classifiers process_images

top10: top10.cpp

find_errors: find_errors.cpp compute_threshold.cpp
	@$(CXX) -Wall -O2 -o $@ $+

precompute_feature_values: precompute_feature_values.cpp compute_feature.cpp
	@$(CXX) -Wall -O2 -o $@ $+

weak_classifiers: weak_classifiers.cpp

process_images: process_images.cpp

%: %.cpp
	@echo "CXX $<"
	@$(CXX) -Wall -O2 -o $@ $<

clean:
	@rm -f  *.o
	@rm -rf *.dSYM

.PHONY: all clean
