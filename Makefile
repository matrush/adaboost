# all: weak_classifiers process_images

precompute_feature_values: precompute_feature_values.o compute_feature.o utils.o
	@$(CXX) -Wall -o $@ $+

weak_classifiers: weak_classifiers.o utils.o
	@$(CXX) -Wall -o $@ $+

process_images: process_images.o utils.o
	@$(CXX) -Wall -o $@ $+

%.o: %.cpp
	@$(CXX) -Wall -O2 -c -o $@ $<

clean:
	@rm -f  *.o
	@rm -rf *.dSYM

.PHONY: all clean
