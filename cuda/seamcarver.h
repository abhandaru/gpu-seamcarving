//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#ifndef __SEAMCARVING_H__
#define __SEAMCARVING_H__

#include <algorithm>
#include <iostream>
#include <vector>

#include "energies.h"
#include "image.h"

class Seamcarver {
 public:
  Seamcarver(Image* image);
  ~Seamcarver();
  void removeSeams(int n);
  void removeSeam();

 private:
  void findSeam();
  float minCost(Energies& energies, int i, int j);
  int nextPower2(int n);

  // data
  Image* _image;
  std::vector<int> _seam;
};

#endif
