//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#ifndef __ENERGIES_H__
#define __ENERGIES_H__

#include <cmath>

#include "bitmap.h"
#include "image.h"

class Energies {
 public:
  Energies(Image* image);
  ~Energies();

  // filter options
  void gradient();

  // getters
  size_t width() const;
  size_t height() const;
  float get(int row, int col) const;
  const float* operator [](int i) const;

 private:
  // data
  Image* _image;
  size_t _width;
  size_t _height;
  float* _energies;
};

#endif