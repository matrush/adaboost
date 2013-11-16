# all: weak_classifiers process_images

top10: top10.cpp

adaboost: adaboost.cpp compute_threshold.cpp compute_error.cpp compute_feature.cpp
	@$(CXX) -Wall -O2 -o $@ $+

find_errors: find_errors.cpp compute_threshold.cpp compute_error.cpp
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
	@rm -f top10 find_errors precompute_feature_values weak_classifiers process_images

.PHONY: all clean
