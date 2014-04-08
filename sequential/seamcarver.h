//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#ifndef __SEAMCARVING_H__
#define __SEAMCARVING_H__

#include <iostream>

#include "energies.h"
#include "image.h"


class Seamcarver {
 public:
  Seamcarver(Image* image);
  ~Seamcarver();
  void removeSeam();

 private:
  // data
  Image* _image;
};

#endif
